
### importing important libraries
import pdb
import sys
import argparse
import torchvision
from utilsDA import *
sys.path.append('../models/')
from models import *
import models
import wrn

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

def main():

    manualSeed = 1  # random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()

    # optional argumentsn
    parser.add_argument("--experiment_name"     , default='mnist_to_svhn_'            , type=str      , help="give experiment a name to keep track of results")
    parser.add_argument("--gpu"                 , default=True               , type=str2bool , help="change to False if you want to train on CPU (Seriously??)")
    parser.add_argument("--lr"                  , default=0.001           , type=float    , help="set learning rate (default recommended)")
    parser.add_argument("--source"          , default='mnist'            , type=str , help="train dataset")
    parser.add_argument("--target"          , default='mnist'            , type=str , help="train dataset")

    parser.add_argument("--method"          , default='dd'            , type=str , help="train dataset")
    parser.add_argument("--iterations"          , default=10000              , type=float    , help="desired iterations for training (not to be confused with epoch)")
    parser.add_argument("--classname"          , default=0            , type=int   , help="desired iterations for training (not to be confused with epoch)")
    args = parser.parse_args()
    from parameters import hyper_para
    '''hyper_para.gpu						= args.gpu
    hyper_para.lr 						= args.lr
    hyper_para.iterations				= args.iterations'''
    #hyper_para.method = 'dd' #'balancedsourcetarget'#'justsource'
    hyper_para.target = args.target
    hyper_para.source = args.source
    hyper_para.method = args.method
    hyper_para.classname = args.classname
    hyper_para.experiment_name 	= args.experiment_name + hyper_para.method  #args.experiment_name
    hyper_para.inclass = [int(hyper_para.classname)]#[0,3,5,7,8,9]
    print(hyper_para.inclass)
    hyper_para.source_n = 2000
    hyper_para.target_n = 5
    #C = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10).cuda()
    #hyper_para.C = C
    #Res = os_test_ens(None,hyper_para,None,True)

    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    octrain(hyper_para)

    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    octest0(hyper_para)

    '''normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)
    testTransform = transforms.Compose([ transforms.ToTensor(), normTransform])
    mnist = torchvision.datasets.CIFAR10('../', train=False, transform=testTransform,  download=True)
    testLoader = DataLoader(mnist, batch_size=64, shuffle=True)
    C = hyper_para.C
    C.load_state_dict(torch.load(hyper_para.experiment_name + '.pth'))
    C.cuda()
    Res = os_test_ens(testLoader,hyper_para,C,True)'''



if __name__ == "__main__":
    main()
