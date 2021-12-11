from numpy.core.fromnumeric import size
from torch._C import device
from utils import *
from data_loader import *

# sys.path.append('./')
from model.models import *
import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from getnc_data import build_data



class Runner(object):

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)
        
        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits

        """

        ent_set, rel_set = OrderedSet(), OrderedSet()
        (
            A,
            X,
            Trip,
            y,
            labeled_nodes_idx,
            train_idx,
            test_idx,
            rel_dict,
            node_dict,
            train_names,
            test_names,
        ) = build_data(self.p.dataset, K=1)
        (
            self.y_train,
            self.y_val,
            self.y_test,
            self.idx_train,
            self.idx_val,
            self.idx_test,
        ) = get_splits(y, train_idx, test_idx, True)
        self.p.num_class = y.shape[1]
        self.labels = torch.LongTensor(np.array(np.argmax(y, axis=-1)).squeeze()).to(self.device)
        self.all_triples = Trip

        self.ent2id = node_dict
        self.rel2id = rel_dict
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for rel, idx in rel_dict.items()})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent		= len(self.ent2id)
        self.p.num_rel		= len(self.rel2id) // 2

        self.triple_data = []
        for sub, rel, obj in self.all_triples:
            sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
            self.triple_data.append((sub, rel, obj))

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------
        
        Returns
        -------
        Constructs the adjacency matrix for GCN
        
        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.triple_data:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.triple_data:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index	= torch.LongTensor(edge_index).to(self.device).t()
        edge_type	= torch.LongTensor(edge_type). to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model
        
        Returns
        -------
        Creates computational graph and optimizer
        
        """
        self.p			= params
        self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model        = self.add_model(self.p.model, self.p.score_func)
        self.optimizer    = self.add_optimizer(self.model.parameters())
        self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
        # self.bceloss	= torch.nn.BCELoss()


    def add_model(self, model, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created
        
        Returns
        -------
        Creates the computational graph for model and initializes it
        
        """
        model_name = '{}_{}'.format(model, score_func)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_type = self.edge_type.to(self.device)
        model = GGPNForEntityClassification(self.edge_index, self.edge_type, params=self.p)
        # torch.distributed.init_process_group(backend="nccl")
        # torch.cuda.set_device(args.local_rank)
        # model = torch.nn.DataParallel(model)
        model.to(self.device)
        # model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
        # model.to(self.device)
        return model

    def add_optimizer(self, parameters):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model
        
        Returns
        -------
        Returns an optimizer for learning the parameters of the model
        
        """
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)


    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved
        
        Returns
        -------
        """
        state = {
            'state_dict'	: self.model.state_dict(),
            'best_val'	: self.best_val,
            'best_epoch'	: self.best_epoch,
            'optimizer'	: self.optimizer.state_dict(),
            'args'		: vars(self.p)
        }
        if args.local_rank == 0:
            torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model
        
        Returns
        -------
        """
        # torch.distributed.barrier()
        # print(load_path)
        state			= torch.load(load_path)
        # state			= torch.load(load_path)
        state_dict		= state['state_dict']
        self.best_val		= state['best_val']
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def compute_loss(self, pred, true_label):
        index = true_label.unsqueeze(1)
        true_label = torch.zeros(pred.size(0), self.p.num_class).to(self.device)
        assert not torch.isnan(true_label).any()
        true_label = true_label.scatter_(1, index, 1)
        true_label = (1.0 - self.p.lbl_smooth)*true_label + (self.p.lbl_smooth/self.p.num_class)
        self.celoss = torch.nn.CrossEntropyLoss()
        bceloss = torch.nn.BCELoss()
        return bceloss(pred, true_label)

    def train(self, epoch):
        # save_path = os.path.join('./checkpoints', self.p.name)
        # t = time.time()
        # Start training
        self.model.train()
        emb_train = self.model.forward()
        # print(emb_train)
        # print(len(self.idx_train))
        loss = self.compute_loss(emb_train[self.idx_train], self.labels[self.idx_train])
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(
            "Epoch: {epoch}, Training Loss on {num} training data: {loss}".format(
                epoch=epoch, num=len(self.idx_train), loss=str(loss.item())
            )
        )
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
                # self.model_state = {
                #     "state_dict": self.model.state_dict(),
                #     "best_val": acc_val,
                #     "best_epoch": epoch,
                #     "optimizer": self.optimizer.state_dict(),
                # }
        return loss
    def evaluate(self, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count
        
        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        with torch.no_grad():
            self.model.eval()
            emb_valid = self.model()
            loss_val = self.compute_loss(
                emb_valid[self.idx_test], self.labels[self.idx_test]
            )
            acc_val = accuracy(emb_valid[self.idx_test], self.labels[self.idx_test])
            # if acc_val >= self.best_val:
            #     self.best_val = acc_val
            #     self.save_model(save_path)
            print(
                "loss_val: {:.4f}".format(loss_val),
                "acc_val: {:.4f}".format(acc_val),
                "time: {:.4f}s".format(time.time()),
            )
            print("\n")
            return  loss_val, acc_val

    def test(self):
        with torch.no_grad():
            self.model.eval()
            emb_test = self.model()
            loss_test = self.compute_loss(
                emb_test[self.idx_test], self.labels[self.idx_test]
            )
            acc_test = accuracy(emb_test[self.idx_test], self.labels[self.idx_test])
            print(
                "Accuracy of the network on the {num} test data: {acc} %, loss: {loss}".format(
                    num=len(self.idx_test), acc=acc_test * 100, loss=loss_test
                )
            )
    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------
        
        Returns
        -------
        """
        save_path = os.path.join('./checkpoints', self.p.name)
        self.best_val = 0.
        self.best_epoch = -1
        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss = self.train(epoch)
            val_loss, val_acc = self.evaluate(epoch)
            if val_acc >= self.best_val:
                kill_cnt = 0
                self.best_val = val_acc
                self.best_epoch = epoch
                self.save_model(save_path)
                print(
                    "loss_val: {:.4f}".format(val_loss),
                    "acc_val: {:.4f}".format(val_acc),
                    "time: {:.4f}s".format(time.time()),
                )
                print("\n")
            else:
                kill_cnt += 1
                if kill_cnt > 500: 
                    self.logger.info("Early Stopping!!")
                    break
            self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid Loss: {:.5}, Valid ACC: {:.5}\n\n'.format(epoch, train_loss, val_loss, self.best_val))
        # torch.distributed.barrier()
        self.load_model(save_path)
        self.test()


if __name__ == '__main__':
    """
        mutag : python run_node_classification.py -hid_layer 3 -init_dim 100 -gcn_dim 100 -gcn_layer 2 -lr 0.001  -gcn_drop 0.1

    """
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
    parser.add_argument('-data',		dest='dataset',         default='mutag',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')
    parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
    parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

    parser.add_argument('-batch',           dest='batch_size',      default=256,    type=int,       help='Batch size')
    parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
    parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=200,  	help='Number of epochs')
    parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
    parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
    parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

    parser.add_argument('-restore',         dest='restore',    action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

    parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim',	dest='embed_dim', 	default=100,   type=int, 	help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=2,   	type=int, 	help='Number of GCN Layers to use')
    parser.add_argument('-hid_layer',	dest='hid_layer', 	default=2,   	type=int, 	help='Number of Hidden Layers to use')
    parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')
    parser.add_argument('-rff_samples',  	dest='rff_samples', 	default=100,  	type=int,	help='Number of RFF features')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
    parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
    parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
    parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')

    parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
    parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
    parser.add_argument("--local_rank",  type=int,  default=0,  help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
    set_gpu(args.gpu)
    # torch.distributed.init_process_group(backend="nccl",init_method='file:///tmp/somefile', rank=0, world_size=1)
    # torch.cuda.set_device(args.local_rank)
    # rank = torch.distributed.get_rank()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = Runner(args)
    model.fit()
