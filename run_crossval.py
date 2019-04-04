import sys
import os
sys.path.insert(0, os.path.abspath('lib/'))


from dict_recursive_update import recursive_update
from socket import gethostname

from niclib.experiment import NIC_Experiment

from niclib.dataset.isles15siss import Isles2015_SISS
from niclib.dataset.isles15spes import Isles2015_SPES

from niclib.architecture.SUNet import SUNETx4


from niclib.patch.sampling import *

from niclib.network.generator import *
from niclib.network.optimizers import TorchOptimizer
from niclib.network.loss_functions import *
from niclib.network.training import EarlyStoppingTrain

from niclib.evaluation.prediction import PatchPredictor
from niclib.evaluation.crossvalidation import SimpleCrossvalidation
from niclib.evaluation.testing import TestingPrediction

from niclib.io.results import load_result_set
from niclib.postprocessing.binarization import thresh_size_search, compute_set_metrics_dict
from niclib.io.metrics import *

from niclib.utils import *

torch.set_default_tensor_type('torch.FloatTensor')

def run_experiment_queue():
    bar = '\n' + '=' * 125
    print(bar + bar + "\n{:-^125}".format("   Running queued experiments   ") + bar + bar, sep='')

    config_queue = [
        {'name': 'crossval_SISS', 'params': {
            'dataset': Isles2015_SISS(dataset_path='~/datasets/ISLES/ISLES2015_SISS', symmetric_modalities=True),
            'architecture': SUNETx4(in_ch=4 * 2, out_ch=2, nfilts=32, ndims=3)
        }},
        {'name': 'crossval_SPES', 'params': {
            'dataset': Isles2015_SPES(dataset_path='~/datasets/ISLES/ISLES2015_SPES', symmetric_modalities=True),
            'architecture': SUNETx4(in_ch=7 * 2, out_ch=2, nfilts=32, ndims=3)
        }},
    ]


    N, n = len(config_queue), 0
    while len(config_queue) > 0:
        config = config_queue.pop(0)
        n += 1
        run_name = check_filename_chars(config['name'])
        bar = '\n{}\n'.format('=' * 75)
        print(bar + "Running experiment {}/{}:\n{}: {}".format(n, N-1, run_name, config['params']) + bar, sep='')
        run_experiment(run_name, config['params'])


def run_experiment(run_name, params=None):
    exp = NIC_Experiment('stroke_mri_segmentation', run_name, device_name='cuda')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("Using GPU {}".format(os.environ['CUDA_VISIBLE_DEVICES']))

    # --------------------------------------------------------------
    #    CONFIGURATION
    # --------------------------------------------------------------
    cfg = {
        # Dataset
        'dataset': None,
        'architecture': None,

        # Patches
        'npatches': 10000,
        'patch_shape': (24, 24, 16),
        'ratio_lesion': 0.5,

        # Training
        'do_train': True,
        'batch_size': 16,
        'train_loss': NIC_binary_focal_loss(gamma=2.0, alpha=0.25),
        'early_stop': {'patience': 8, 'metric': 'l1_er'},

        # Evaluation
        'num_folds': 4,
        'crossval_folds': None,

        # Prediction
        'patch_shape_pred': (24, 24, 16),
        'pred_extraction_step': (4, 4, 1),
        'uncertainty': {'runs': 0, 'rate': 0.1, 'type': 'alpha'}
    }
    if params is not None:
        print("Setting parameters in configuration queue")
        recursive_update(cfg, params)  # Overwrite cfg with keys present in paramas (others left untouched)

    # Other parameters
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lesion_sizes = [1, 10, 50, 100, 200, 300, 500, 1000]

    # --------------------------------------------------------------
    #    Experiment
    # --------------------------------------------------------------

    # 1. Dataset
    assert isinstance(cfg['dataset'], NIC_Dataset)

    dataset = copy.copy(cfg['dataset'])
    dataset.load()
    print("Loaded {} dataset with {} training and {} testing".format(
        dataset.__class__.__name__, len(dataset.train), len(dataset.test)))

    # 2. Model
    model_def = copy.deepcopy(cfg['architecture'])

    model_parameters = filter(lambda p: p.requires_grad, model_def.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    print("Setting model instance from {} architecture with {} parameters".format(model_def.__class__.__name__, nparams))

    # 3. Training and validation sets
    train_gen = PatchGeneratorBuilder(
        batch_size=cfg['batch_size'],
        zeropad_shape=cfg['patch_shape'],
        instruction_generator=PatchInstructionGenerator(
            in_shape=cfg['patch_shape'],
            out_shape=cfg['patch_shape'],
            sampler=StrokeLesionSampling(
                in_shape=cfg['patch_shape'], num_patches=cfg['npatches'], ratio_lesion=cfg['ratio_lesion'], augment_factor=2),
            augment_to=cfg['npatches']),
        num_workers=4,
        shuffle=True)

    val_gen = PatchGeneratorBuilder(
        batch_size=cfg['batch_size'],
        zeropad_shape=cfg['patch_shape'],
        instruction_generator=PatchInstructionGenerator(
            in_shape=cfg['patch_shape'],
            out_shape=cfg['patch_shape'],
            sampler=StrokeLesionSampling(
                in_shape=cfg['patch_shape'], num_patches=cfg['npatches'], ratio_lesion=cfg['ratio_lesion'], augment_factor=2),
            augment_to=cfg['npatches']),
        num_workers=4,
        shuffle=True)

    trainer = EarlyStoppingTrain(
        do_train=cfg['do_train'],
        device=exp.get_device(),
        max_epochs=200,
        batch_size=cfg['batch_size'],
        loss_func=cfg['train_loss'],
        optimizer=TorchOptimizer(torch.optim.Adadelta, opts={'rho':0.95}),
        train_metrics={'acc':nic_binary_accuracy},
        eval_metrics={
            'acc': nic_binary_accuracy,
            'l1_er': nic_binary_l1_er},
        early_stopping_metric=cfg['early_stop']['metric'],
        early_stopping_patience=cfg['early_stop']['patience'])

    predictor = PatchPredictor(
        device=exp.get_device(),
        num_classes=2,
        lesion_class=1,
        uncertainty_passes=cfg['uncertainty']['runs'],
        uncertainty_dropout=cfg['uncertainty']['rate'],
        uncertainty_dotype=cfg['uncertainty']['type'],
        zeropad_shape = cfg['patch_shape_pred'],
        instruction_generator=PatchGeneratorBuilder(
            batch_size=64,
            shuffle=False,
            zeropad_shape=None,
            instruction_generator=PatchInstructionGenerator(
                in_shape=cfg['patch_shape_pred'],
                out_shape=cfg['patch_shape_pred'],
                sampler=UniformSampling(
                    in_shape=cfg['patch_shape_pred'], extraction_step=cfg['pred_extraction_step']),
                augment_to=None)))

    crossval = SimpleCrossvalidation(
        model_definition=model_def,
        images=dataset.train,
        num_folds=cfg['num_folds'],
        model_trainer=trainer,
        train_instr_gen=train_gen,
        val_instr_gen=val_gen,
        checkpoint_pathfile=os.path.join(exp.checkpoints_dir, '{}.pt'.format(run_name)),  # '{}.pt'.format(run_name)
        log_pathfile=exp.get_log_filename(),
        test_predictor=predictor,
        results_path=os.path.join(exp.results_dir, '{}/'.format(run_name)),
    )

    # EXECUTION

    # Run validation
    crossval.run_crossval()

    # Load probability maps
    results = load_result_set(os.path.join(exp.results_dir, '{}/'.format(run_name)), dataset.train, result_type='pred', file_format='nii.gz')

    # Grid search of post-processing options
    metrics_list, metrics_names = thresh_size_search(results, dataset.train, thresholds, lesion_sizes)
    print_metrics_list(metrics_list, metrics_names)
    save_metrics_list(os.path.join(exp.metrics_dir, '{}_{}.csv'.format(get_formatted_timedate(), run_name)), metrics_list, metrics_names)


if __name__ == '__main__':
    run_experiment_queue()
