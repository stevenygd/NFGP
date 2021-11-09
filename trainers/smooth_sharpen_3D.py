import os
from trainers.smooth_sharpen import Trainer as BaseTrainer
from trainers.utils.vis_utils import imf2mesh


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args, original_decoder=original_decoder)
        self.dim = 3

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return
        super().log_train(train_info, train_data, writer=writer,
                          step=step, epoch=epoch, visualize=visualize, **kwargs)

        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v)
                      for k, v in train_info.items()}
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            if step is not None:
                writer.add_scalar('train/' + k, v, step)
            else:
                assert epoch is not None
                writer.add_scalar('train/' + k, v, epoch)

        if self.show_network_hist:
            for name, p in self.decoder.named_parameters():
                if step is not None:
                    writer.add_histogram("hist/%s" % name, p, step)
                else:
                    assert epoch is not None
                    writer.add_histogram("hist/%s" % name, p, epoch)

    def visualize(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        print("Visualize: %s" % step)
        # TODO: use marching cube to save the meshes
        res = int(getattr(self.cfg.trainer, "vis_mc_res", 100))
        thr = float(getattr(self.cfg.trainer, "vis_mc_thr", 0.))
        mesh = imf2mesh(
            lambda x: self.decoder(x, None), res=res, threshold=thr)
        if mesh is not None:
            save_name = "mesh_%diters.obj" % self.num_update_step
            path = os.path.join(self.cfg.save_dir, "val", save_name)
            mesh.export(path)

        # TODO: Sample surface points from the meshes, create a point
        #  cloud to be visualized in matplotlib
        # raise NotImplemented
