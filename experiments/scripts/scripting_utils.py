import torch
from torch.autograd import grad
import clamiter as ci

def print_prior_training_stats(trainer, device):
    
    # CLAM AND PRIOR LOSS FOR ALL
    print(f'prior score for all nodes: \n{trainer.clamiter.prior.forward_ll(trainer.data.x)}')

    print(f'clam loss score for all nodes: \n{ci.clam_loss(trainer.data, lorenz=False)}')
    # ====================================================

    # GRADIENTS
    trainer.data.x.requires_grad = True
    loss = trainer.clamiter.readout(trainer.data)

    gradient = grad(loss, trainer.data.x, create_graph=True)[0].mean()

    print(f'number of features with gradient > 1000:\n {torch.sum(torch.abs(gradient) > 10000)}')
    print(f'maximum gradient:\n {torch.max(torch.abs(gradient), dim=0)}')
    trainer.data.x.requires_grad = False

    print(f'gradient:\n mean: {gradient.mean(dim=0)} ; std: {gradient.std(dim=0)}')
    # ===========================================

    # PRIOR SCORES
    print(f'the loss at the origin:\n {trainer.clamiter.prior.forward_ll(torch.zeros([1, trainer.data.x.shape[1]]).to(device))}')
    prior_score = trainer.clamiter.prior.forward_ll(trainer.data.x, sum=False)
    prior_score_not_far_from = trainer.clamiter.prior.forward_ll(
        trainer.data.x + 0.4, sum=False)
    diff_04 = prior_score - prior_score_not_far_from
    print(f'prior score\n mean: {prior_score.mean()}; std: {prior_score.std()}\n')
    print(f'prior score max: {prior_score.max()} min: {prior_score.min()} \n')
    print(
        f'prior score DIFF mean std (indication to how peaky the function is):\n mean: {(diff_04).mean()} ; std: {(diff_04).std()}')
    print(f'prior score diff per node: \n{diff_04}')

    print(f'\n\nprior score and grad \n\n {prior_score} \n\n gradient: \n\n{gradient}')
    # ===============================================

    #! it does manage to do feat optimization!! maybe there is a problem with fit funciton....
    #todo: how do i know the average distance of features? you can just compare them one time. but actually i want to know the distance of every node to the closest node and then take the as the noise amp... or twice that dunno

    #todo: what i expect to see in the score difference? not really sure, you are going in some direction might be direction of steep decoline....