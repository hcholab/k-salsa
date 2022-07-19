import torch


def get_average_image(net, opts):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    computing_features=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    if opts.dataset_type == "cars_encode":
        avg_image = avg_image[:, 32:224, :]
    return avg_image
    
def run_on_batch(inputs, net, opts, avg_image, computing_features=False, batch_centroids=None, get_latent=False):
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            x_input = torch.cat([inputs, y_hat], dim=1)

        y_hat, latent, feat = net.forward(x_input,
                                    latent=latent,
                                    batch_centroids=batch_centroids,
                                    computing_features=computing_features,
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=opts.resize_outputs)

        if opts.dataset_type == "cars_encode":
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        # store intermediate outputs
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            results_latent[idx].append(latent[idx].cpu().numpy())

        # resize input to 256 before feeding into next iteration
        if opts.dataset_type == "cars_encode":
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)
    if get_latent:
        return y_hat, latent, results_batch, results_latent, feat
    else:
        return latent, results_batch, results_latent, feat

def run_on_centroid(net, opts, nearest_neighbor_centroid=None):
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(nearest_neighbor_centroid.shape[0])}
    results_latent = {idx: [] for idx in range(nearest_neighbor_centroid.shape[0])}
    for iter in range(opts.n_iters_per_batch):
        y_hat, latent, _ = net.forward(x=None,
                            latent=latent,
                            nearest_neighbor_centroid=nearest_neighbor_centroid,
                            randomize_noise=False,
                            return_latents=True,
                            resize=opts.resize_outputs)
        # store intermediate outputs
        for idx in range(nearest_neighbor_centroid.shape[0]):
            results_batch[idx].append(y_hat[idx])
            # results_latent[idx].append(latent[idx].cpu().numpy())
        # resize input to 256 before feeding into next iteration
        if opts.dataset_type == "cars_encode":
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)

    return results_batch, latent
