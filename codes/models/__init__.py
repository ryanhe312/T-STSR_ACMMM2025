import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'VideoSR_base':
        from .VideoSR_base_model import VideoSRBaseModel as M
    elif model == 'VideoSR_base_n2n':
        from .VideoSR_base_model_n2n import VideoSRBaseModel as M
    elif model == 'VideoSR_base_n2n_mixed':
        from .VideoSR_base_model_n2n_mixed import VideoSRBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
