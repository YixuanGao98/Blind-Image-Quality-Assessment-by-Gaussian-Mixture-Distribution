
import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from scipy import stats
import random
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from EMDLoss import EMDLoss
import scipy.stats
from scipy.optimize import curve_fit
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def EMD(y_true, y_pred):
    cdf_ytrue = np.cumsum(y_true, axis=-1)
    cdf_ypred = np.cumsum(y_pred, axis=-1)
    samplewise_emd = np.sqrt(np.mean(np.square(np.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return np.mean(samplewise_emd)

def JSDmetric(y_true, y_pred):
    M=(y_true+y_pred)/2
    js=0.5*scipy.stats.entropy(y_true, M)+0.5*scipy.stats.entropy(y_pred, M)
    return js

def histogram_intersection(h1, h2):
    intersection = 0
    for i in range(len(h1)):
        intersection += min(h1[i], h2[i])
    return intersection
class L1RankLoss(torch.nn.Module):
    
    """
    L1 loss + Rank loss
    """

    def __init__(self, **kwargs):
        super(L1RankLoss, self).__init__()
        self.l1_w = kwargs.get("l1_w", 1)
        self.rank_w = kwargs.get("rank_w", 1)
        self.hard_thred = kwargs.get("hard_thred", 1)
        self.use_margin = kwargs.get("use_margin", False)

    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l1_loss = F.l1_loss(preds, gts) * self.l1_w

        # simple rank
        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
        else:
            rank_loss = masks_hard * torch.relu(- masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_total = l1_loss + rank_loss * self.rank_w
        return loss_total


class MDNPerceptron(nn.Module):
    def __init__(self, in_channels, middle_channels, n_gaussians):
        nn.Module.__init__(self)
        self.n_gaussians = n_gaussians
        # self.histogram=SoftHistogram(n_features=in_channels,n_examples=1,num_bins=5,quantiles=False)
        
        
        self.l1 = self.fc(in_channels, middle_channels, in_channels)

        self.z_pi = self.fc_par_p(in_channels, middle_channels, n_gaussians)
        self.z_mu = self.fc_par_m(in_channels, middle_channels, n_gaussians)
        self.z_sigma = self.fc_par_s(in_channels, middle_channels, n_gaussians)

    """ Returns parameters for a mixture of gaussians given x
    mu - vector of means of the gaussians
    sigma - voctor of the standard deviation of the gaussians
    pi - probability distribution over the gaussians
    """
    def forward(self, x):
        hidden = torch.tanh(self.l1(x))
        # z=self.z_pi(hidden)

        pi = F.softmax(self.z_pi(hidden), 1)
        mu = self.z_mu(hidden)
        sigma = torch.exp(self.z_sigma(hidden))

        return pi, mu, sigma
    
    def fc(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )
        return regression_block
    def fc_par_p(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )
        return regression_block
    def fc_par_m(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )
        return regression_block
    def fc_par_s(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )
        return regression_block
    """Makes a random draw from a randomly selected
    mixture based upon the probabilities in Pi
    """
    def sample(self, pi, mu, sigma):
        mixture = torch.normal(mu, sigma)
        k = torch.multinomial(pi, 1, replacement=True).squeeze()
        result = mixture[range(k.size(0)), k]
        return result

    """Computes the log probability of the datapoint being
    drawn from all the gaussians parametized by the network.
    Gaussians are weighted according to the pi parameter 
    """
    def loss_fn(self, y, pi, mu, sigma):
        mu=mu.unsqueeze(2)
        sigma=sigma.unsqueeze(2)
        pi=pi.unsqueeze(2)
        mixture = torch.distributions.normal.Normal(mu, sigma)
        # b=sigma.size(0)
        y = y.unsqueeze(1)
        c=sigma.size(1)
        y= y.repeat(1,c, 1)
        log_prob = mixture.log_prob(y)
        prob = torch.exp(log_prob)
        weighted_prob = prob * pi
        sum = torch.sum(weighted_prob, dim=1)
        log_prob_loss = -torch.log(sum)
        ave_log_prob_loss = torch.mean(log_prob_loss)
        return ave_log_prob_loss

    
    
    def gmm2hist(self,pi, sigma, mu,y):
        mu=mu.unsqueeze(2)
        sigma=sigma.unsqueeze(2)
        pi=pi.unsqueeze(2)
        mixture = torch.distributions.normal.Normal(mu, sigma)
        b=sigma.size(0)
        c=sigma.size(1)
        y= y.repeat(b,c, 1)

        log_prob = mixture.log_prob(y)
        prob = torch.exp(log_prob)
        weighted_prob = prob * pi
        sum = torch.sum(weighted_prob, dim=1)
        return sum

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs
    def forward(self, x,x_gray):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x_gray)
        x2 = self.conv1_2(x_gray)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)
from model import MANIQA
class GMMIQA(torch.nn.Module):

    def __init__(self, path, options):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.net=MANIQA(path).cuda()
        n_components=2
        self.sample=torch.Tensor([0.1,0.3,0.5,0.7,0.9]).cuda()
        self.n_features =2816*2
        self.numbin=options['numbin']
        self.GMMmodel = MDNPerceptron(in_channels=self.n_features,middle_channels=int(self.n_features/2),  n_gaussians=n_components).cuda()
        self.qr=self.quality_regression(self.n_features,int(self.n_features/2),self.n_features)

        
    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block
    def forward(self, X):
        """Forward pass of the network.
        """

        X1 =self.net(X)

        
        X1 = torch.nn.functional.normalize(X1)
        X1=self.qr(X1)
        pi, mu, sigma=self.GMMmodel(X1)
        samples = self.GMMmodel.gmm2hist(pi, sigma, mu,self.sample)

        hist=F.softmax(samples,dim=1)
        
        return hist,pi, mu, sigma

class main(object):
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path

        # Network.
        self._net = torch.nn.DataParallel(GMMIQA(self._path, self._options), device_ids=[0]).cuda()

        self._criterion1 = EMDLoss()
        self._criterion3 =L1RankLoss()

        # Solver.

        self._solver = torch.optim.Adam(
                self._net.module.parameters(), lr=self._options['base_lr'],
                weight_decay=self._options['weight_decay'])
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self._solver, milestones=[25], gamma=0.1)

        
        resize=512
        crop_size = 384
        train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize,resize)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225))
        ])

            
            
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize,resize)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

           
        if self._options['dataset'] == 'KONIQ10K':
            import KONIQ10KFolder
            train_data = KONIQ10KFolder.KONIQ10KFolder(
                    root=self._path['KONIQ10K'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = KONIQ10KFolder.KONIQ10KFolder(
                    root=self._path['KONIQ10K'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        """Train the network."""
        print('Training.')
        best_Cosine = 0.0
        
        best_MOSsrcc = 0.0
        best_epoch_hist = None
        best_epoch_mos = None
        print('Epoch\tTrain loss\tTest_JSD\tTest_EMD\tTest_RMSE\tTest_inter\tTest_Cosine\t\tTest_MOSsrcc\tTest_MOSplcc\tTest_MOSrmse')
        for t in range(self._options['epochs']):
            epoch_loss = []
            import time
            # time.sleep(0.15)
            for X,gtmos, y in self._train_loader:
                gtmos =gtmos.to(torch.float32)
                # Data.
                X = X.cuda()
                y = y.cuda()
                gtmos = gtmos.cuda()
                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                prehist,pi, mu, sigma = self._net(X)
                ###hist
                loss1 = self._criterion1(prehist, y.view(len(prehist),self._options['numbin']))##predist gtdist loss

                premos=torch.sum(pi*mu,dim=1)*4+1
                premos=premos.unsqueeze(1)
                loss3_pre=self._criterion3(premos, gtmos)#premos gtmos loss
                gtmos=gtmos.unsqueeze(1)

                loss=7*loss1+1*loss3_pre
                
                epoch_loss.append(loss.item())
                loss.requires_grad_(True)
                loss.backward()
                self._solver.step()
                self.scheduler.step()

            JSDtest,EMDtest,RMSEtest,intersectiontest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse = self._consitency(self._test_loader)
            if Cosinetest >= best_Cosine:
                best_Cosine = Cosinetest
                
                best_EMD = EMDtest
                best_RMSE = RMSEtest
                best_inter =intersectiontest
                best_JSD =JSDtest


                best_epoch_hist = t + 1
                print('*', end='')

            if MOSsrcc >= best_MOSsrcc:
                best_MOSsrcc = MOSsrcc
                best_MOSplcc = MOSplcc
                best_MOSrmse = MOSrmse
                best_epoch_mos = t + 1

                

            print('%d\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t+1, sum(epoch_loss) / len(epoch_loss),  JSDtest,EMDtest,RMSEtest,intersectiontest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse))           

        print('Best at epoch %d: test cosine %f, Best at epoch %d: test srcc %f' % (best_epoch_hist, best_Cosine,best_epoch_mos, best_MOSsrcc))
        
        return best_JSD,best_EMD, best_RMSE,best_inter,best_Cosine, best_MOSsrcc,best_MOSplcc,best_MOSrmse

    def _consitency(self, data_loader):
        self._net.train(False)
        # self._net.eval()
        num_total = 0
        JSD_test = []
        JSD_all=0
        JSDtest=0
        JSD0=0
        
        EMD_test = []
        EMD_all=0
        EMDtest=0
        EMD0=0

        RMSE_all=0
        RMSE0=0
        RMSE_test=[]
        RMSEtest=0

        Cosine_all=0
        Cosine0=0
        Cosine_test=[]
        Cosinetest=0

        intersection_test = []
        intersection_all=0
        intersectiontest=0
        pscores_MOS = []
        tscores_MOS = []
        # si = np.arange(1, 6, 1)
        for X,gtmos, y in data_loader:
            # Data.
            X = X.cuda()
            y = y.cuda()
            gtmos= gtmos.cuda()
            # Prediction.
            prehist,pi, mu, sigma= self._net(X)
            score=prehist
            score=score[0].cpu()
            y=y[0].cpu()
            gtmos=gtmos[0].cpu()

            calmos=torch.sum(pi*mu)*4+1
            pscores_MOS.append(calmos.cpu().detach().numpy())
            tscores_MOS.append(gtmos.detach().numpy())
            
            ##histogram
            RMSE0=np.sqrt(((score.detach().numpy() - y.detach().numpy()) ** 2).mean())#对于每张直方图，求结果
            EMD0=EMD(score.detach().numpy(),y.detach().numpy())
            JSD0=JSDmetric(score.detach().numpy(),y.detach().numpy())
            intersection0=histogram_intersection(score.detach().numpy(),y.detach().numpy())
            X=[score.detach().numpy(),y.detach().numpy()]
            Cosine0 = (1-pairwise_distances( X, metric='cosine'))[0][1]
            JSD_test.append(JSD0)
            EMD_test.append(EMD0)
            RMSE_test.append(RMSE0)
            intersection_test.append(intersection0)
            Cosine_test.append(Cosine0)
        tscores_MOS=np.squeeze(tscores_MOS)
        pscores_MOS=np.squeeze(pscores_MOS)
        num_total =len(EMD_test)
        for ele in range(0, len(EMD_test)):
            JSD_all = JSD_all + JSD_test[ele]  
            EMD_all = EMD_all + EMD_test[ele]  
            RMSE_all = RMSE_all + RMSE_test[ele]  
            intersection_all = intersection_all + intersection_test[ele] 
            Cosine_all = Cosine_all + Cosine_test[ele] 
        JSDtest=JSD_all/num_total
        EMDtest=EMD_all/num_total
        RMSEtest=RMSE_all/num_total
        intersectiontest=intersection_all/num_total
        Cosinetest=Cosine_all/num_total
        
        ##MOS
        pscores_MOS_logistic = fit_function(tscores_MOS,pscores_MOS)
        MOSsrcc, _ = stats.spearmanr(pscores_MOS_logistic,tscores_MOS)
        MOSplcc, _ = stats.pearsonr(pscores_MOS_logistic,tscores_MOS)
        MOSrmse=np.sqrt((((pscores_MOS_logistic)-np.array(tscores_MOS))**2).mean())
        self._net.train(True)  # Set the model to training phase
        return JSDtest,EMDtest,RMSEtest,intersectiontest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse

# def main():
if __name__ == '__main__':
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train GMM for BIQA.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-4,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=8, help='Batch size:8.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=50, help='Epochs for training:50.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-12, help='Weight decay.')
    parser.add_argument('--dataset',dest='dataset',type=str,default='KONIQ10K',
                        help='dataset: live|KONIQ10K')
    parser.add_argument('--dataset_path',  type=str, default='/mnt/sda/gyx/image_database/KON')
    parser.add_argument("--swin_model", type=str, default='/home/gyx/.cache/torch/hub/checkpoints/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth')
    parser.add_argument("--vision_tower_name", type=str, default='/mnt/sda/gyx/huggingface/clip-vit-large-patch14-336')

    args = parser.parse_args()

    seed = random.randint(10000000, 99999999)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("seed:", seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset':args.dataset,
        'fc': [],
        'train_index': [],
        'test_index': [],
        
        'train_num': 0,
        'numbin':5
    }
    
    path = {
        'KONIQ10K':args.dataset_path,
        'swin_model': args.swin_model,
        'vision_tower_name': args.vision_tower_name,
    }
    
    if options['dataset'] == 'KONIQ10K':          
        index = list(range(0,10073))
        options['numbin'] == 5   
    
    
    
    lr_backup = options['base_lr']
    EMD_all = np.zeros((1,10),dtype=np.float64)
    RMSE_all = np.zeros((1,10),dtype=np.float64)
    Cosine_all = np.zeros((1,10),dtype=np.float64)
    JSD_all = np.zeros((1,10),dtype=np.float64)
    inter_all = np.zeros((1,10),dtype=np.float64)
    
    MOSsrcc_all = np.zeros((1,10),dtype=np.float64)
    MOSplcc_all = np.zeros((1,10),dtype=np.float64)
    MOSrmse_all = np.zeros((1,10),dtype=np.float64)
    
    for i in range(0,10):
        #randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(0.8*len(index))]
        test_index = index[round(0.8*len(index)):len(index)]
        options['train_num'] = i
        options['train_index'] = train_index
        options['test_index'] = test_index

        #fine-tune all model
        options['fc'] = False
        options['base_lr'] = lr_backup
        manager = main(options, path)
        JSD,EMD, RMSE,inter,Cosine, MOSsrcc,MOSplcc,MOSrmse = manager.train()
        
        EMD_all[0][i] = EMD
        RMSE_all[0][i] = RMSE
        Cosine_all[0][i] = Cosine
        JSD_all[0][i] = JSD
        inter_all[0][i] = inter
        
        #
        MOSsrcc_all[0][i] = MOSsrcc
        MOSplcc_all[0][i] = MOSplcc
        MOSrmse_all[0][i] = MOSrmse
        print(i)
        
        

        
        
    EMD_mean = np.mean(EMD_all)
    RMSE_mean = np.mean(RMSE_all)
    Cosine_mean = np.mean(Cosine_all)
    JSD_mean = np.mean(JSD_all)
    inter_mean = np.mean(inter_all)
    
    MOSsrcc_mean = np.mean(MOSsrcc_all)
    MOSplcc_mean = np.mean(MOSplcc_all)
    MOSrmse_mean = np.mean(MOSrmse_all)
    

    


    # srcc_mean = np.mediam(srcc_all)
    print('n_components=2'+'\n')
    print('loss=7*loss1+1*loss3_pre'+'\n')
    print("seed:", seed)
    print(JSD_all)
    print('average JSD:%4.4f' % (JSD_mean))  
    print(EMD_all)
    print('average EMD:%4.4f' % (EMD_mean))  
    print(RMSE_all)
    print('average RMSE:%4.4f' % (RMSE_mean))  
    print(inter_all)
    print('average inter:%4.4f' % (inter_mean))  
    print(Cosine_all)
    print('average Cosine:%4.4f' % (Cosine_mean))  
    
    
    print(MOSsrcc_all)
    print('average Cosine:%4.4f' % (MOSsrcc_mean))  
    print(MOSplcc_all)
    print('average Cosine:%4.4f' % (MOSplcc_mean))  
    print(MOSrmse_all)
    print('average Cosine:%4.4f' % (MOSrmse_mean))  
    
    # return EMD_all,RMSE_all,Cosine_all,MOSsrcc_all,MOSplcc_all,MOSrmse_all


# if __name__ == '__main__':
    # main()
