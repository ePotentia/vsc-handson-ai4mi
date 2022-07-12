#https://melhanna.com/fastai-explainability-with-shap/
import shap 
import matplotlib.pyplot as pl
from shap.plots import colors
import numpy as np 

def image_shap(learn,test_dls):
    batch = next(iter(test_dls.train))
    images, _ = batch
    num_samples = 8
    background = images[:num_samples]
    test_images = images[num_samples:]

    e = shap.GradientExplainer(learn.model.cuda(), background.cuda())
    shap_values = e.shap_values(test_images.cuda())
    
    for idx, x in enumerate(batch[0][num_samples:num_samples+8]):
        x = x.cpu() # move image to CPU
        label = test_dls.train.vocab[batch[1][num_samples:num_samples+8]][idx]
        sv_idx = list(test_dls.train.vocab).index(label)

        # plot our explanations
        fig, axes = pl.subplots(figsize=(7, 7))

        # make sure we have a 2D array for grayscale
        if len(x.shape) == 3 and x.shape[2] == 1:
            x = x.reshape(x.shape[:2])
        if x.max() > 1:
            x /= 255.

        # get a grayscale version of the image
        x_curr_gray = (
            0.2989 * x[0,:,:] +
            0.5870 * x[1,:,:] +
            0.1140 * x[2,:,:]
        )
        x_curr_disp = x

        abs_vals = np.stack(
            [np.abs(shap_values[sv_idx][idx].sum(0))], 0
        ).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)

        label_kwargs = {'fontsize': 12}
        axes.set_title(label, **label_kwargs)

        sv = shap_values[sv_idx][idx].sum(0)
        axes.imshow(
            x_curr_gray,
            cmap=pl.get_cmap('gray'),
            alpha=0.7,
            extent=(-1, sv.shape[1], sv.shape[0], -1)
        )
        im = axes.imshow(
            sv,
            cmap=colors.red_transparent_blue, 
            vmin=-max_val, 
            vmax=max_val
        )
        axes.axis('off')

        fig.tight_layout()

        cb = fig.colorbar(
            im, 
            ax=np.ravel(axes).tolist(),
            label="SHAP value",
            orientation="horizontal"
        )
        cb.outline.set_visible(False)
        pl.show()
