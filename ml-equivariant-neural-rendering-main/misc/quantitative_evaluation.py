import torch
import torch.nn.functional as F
from misc.dataloaders import create_batch_from_data_list


def get_dataset_psnr(device, model, dataset, source_img_idx_shift=64,
                     batch_size=10, max_num_scenes=None):
    """Returns PSNR for each scene in a dataset by comparing the view predicted
    by a model and the ground truth view.

    Args:
        device (torch.device): Device to perform PSNR calculation on.
        model (models.neural_renderer.NeuralRenderer): Model to evaluate.
        dataset (misc.dataloaders.SceneRenderDataset): Dataset to evaluate model
            performance on. Should be one of "chairs-test" or "cars-test".
        source_img_idx_shift (int): Index of source image for each scene. For
            example if 00064.png is the source view, then
            source_img_idx_shift = 64.
        batch_size (int): Batch size to use when generating predictions. This
            should be a divisor of the number of images per scene.
        max_num_scenes (None or int): Optionally limit the maximum number of
            scenes to calculate PSNR for.

    Notes:
        This function should be used with the ShapeNet chairs and cars *test*
        sets.
    """
    num_imgs_per_scene = dataset.num_imgs_per_scene
    num_scenes = dataset.num_scenes

    if max_num_scenes is not None:
        num_scenes = min(max_num_scenes, num_scenes)

    # Ensure batch size divides image count per scene
    assert (num_imgs_per_scene - 1) % batch_size == 0, \
        f"Batch size {batch_size} must divide number of images per scene {num_imgs_per_scene}."

    batches_per_scene = (num_imgs_per_scene - 1) // batch_size
    psnrs = []

    for i in range(num_scenes):
        source_img_idx = i * num_imgs_per_scene + source_img_idx_shift

        # ✅ Prevent out-of-range indexing
        if source_img_idx >= len(dataset):
            break

        img_source = dataset[source_img_idx]["img"].unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        render_params = dataset[source_img_idx]["render_params"]
        azimuth_source = torch.Tensor([render_params["azimuth"]]).repeat(batch_size).to(device)
        elevation_source = torch.Tensor([render_params["elevation"]]).repeat(batch_size).to(device)

        scenes = model.inverse_render(img_source)

        num_points_in_batch = 0
        data_list = []
        scene_psnr = 0.0

        for j in range(num_imgs_per_scene):
            # ✅ Secondary safeguard for final edge-case scenes
            target_idx = i * num_imgs_per_scene + j
            if target_idx >= len(dataset):
                break

            if j == source_img_idx_shift:
                continue

            data_list.append(dataset[target_idx])
            num_points_in_batch += 1

            if num_points_in_batch == batch_size:
                img_target, azimuth_target, elevation_target = create_batch_from_data_list(data_list)
                img_target = img_target.to(device)
                azimuth_target = azimuth_target.to(device)
                elevation_target = elevation_target.to(device)

                rotated = model.rotate_source_to_target(
                    scenes, azimuth_source, elevation_source, azimuth_target, elevation_target
                )
                img_predicted = model.render(rotated).detach()
                scene_psnr += get_psnr(img_predicted, img_target)

                data_list = []
                num_points_in_batch = 0

        psnrs.append(scene_psnr / batches_per_scene)

        print("{}/{}: Current - {:.3f}, Mean - {:.4f}".format(
            i + 1, num_scenes, psnrs[-1], torch.mean(torch.Tensor(psnrs))
        ))

    return psnrs


def get_psnr(prediction, target):
    """Returns PSNR between a batch of predictions and a batch of targets."""
    batch_size = prediction.shape[0]
    mse_per_pixel = F.mse_loss(prediction, target, reduction='none')
    mse_per_img = mse_per_pixel.view(batch_size, -1).mean(dim=1)
    psnr = 10 * torch.log10(1 / mse_per_img)
    return torch.mean(psnr).item()
