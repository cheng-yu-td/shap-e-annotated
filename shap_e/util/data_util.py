import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional, Union

import blobfile as bf
import numpy as np
import torch
from PIL import Image

from shap_e.rendering.blender.render import render_mesh, render_model
from shap_e.rendering.blender.view_data import BlenderViewData
from shap_e.rendering.mesh import TriMesh
from shap_e.rendering.point_cloud import PointCloud
from shap_e.rendering.view_data import ViewData
from shap_e.util.collections import AttrDict
from shap_e.util.image_util import center_crop, get_alpha, remove_alpha, resize


# Define a function that creates a batch of multimodal data. This can be used for tasks like training a machine learning model.
def load_or_create_multimodal_batch(
        device: torch.device,
        *,
        mesh_path: Optional[str] = None,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        point_count: int = 2 ** 14,
        random_sample_count: int = 2 ** 19,
        pc_num_views: int = 40,
        mv_light_mode: Optional[str] = None,
        mv_num_views: int = 20,
        mv_image_size: int = 512,
        mv_alpha_removal: str = "black",
        verbose: bool = False,
) -> AttrDict:
    # This function creates point cloud data, multiple views of a 3D model (if specified), and normalizes the data.

    if verbose:  # If verbose mode is activated, print progress messages.
        print("creating point cloud...")

    # Load or create point cloud from either a mesh file or a pre-trained 3D model.
    pc = load_or_create_pc(
        mesh_path=mesh_path,
        model_path=model_path,
        cache_dir=cache_dir,
        random_sample_count=random_sample_count,
        point_count=point_count,
        num_views=pc_num_views,
        verbose=verbose,
    )

    # Concatenate point coordinates and color channels.
    raw_pc = np.concatenate([pc.coords, pc.select_channels(["R", "G", "B"])], axis=-1)

    # Convert the numpy array to a PyTorch tensor, move it to the specified device and store it in the batch.
    encode_me = torch.from_numpy(raw_pc).float().to(device)
    batch = AttrDict(points=encode_me.t()[None])

    if mv_light_mode:  # If a lighting mode for the multiview is specified, create multiview data.
        if verbose:
            print("creating multiview...")
        with load_or_create_multiview(
                mesh_path=mesh_path,
                model_path=model_path,
                cache_dir=cache_dir,
                num_views=mv_num_views,
                extract_material=False,
                light_mode=mv_light_mode,
                verbose=verbose,
        ) as mv:
            # Create empty lists to store multiview data.
            cameras, views, view_alphas, depths = [], [], [], []

            for view_idx in range(mv.num_views):  # For each view in the multiview data:
                # Load the view data. Channels include Red, Green, Blue, and potentially Alpha.
                camera, view = mv.load_view(
                    view_idx,
                    ["R", "G", "B", "A"] if "A" in mv.channel_names else ["R", "G", "B"],
                )

                depth = None
                if "D" in mv.channel_names:  # If depth channel is available:
                    _, depth = mv.load_view(view_idx, ["D"])  # Load depth data.
                    depth = process_depth(depth, mv_image_size)  # Process the depth data.

                # Process the view image, handle alpha channel and resize the image.
                view, alpha = process_image(
                    np.round(view * 255.0).astype(np.uint8), mv_alpha_removal, mv_image_size
                )

                # Process the camera image, center crop and resize it.
                camera = camera.center_crop().resize_image(mv_image_size, mv_image_size)

                # Add processed data to the corresponding lists.
                cameras.append(camera)
                views.append(view)
                view_alphas.append(alpha)
                depths.append(depth)

            # Add the lists to the batch.
            batch.depths = [depths]
            batch.views = [views]
            batch.view_alphas = [view_alphas]
            batch.cameras = [cameras]

    # Normalize the batch data and return it.
    return normalize_input_batch(batch, pc_scale=2.0, color_scale=1.0 / 255.0)


def load_or_create_pc(
        *,
        mesh_path: Optional[str],
        model_path: Optional[str],
        cache_dir: Optional[str],
        random_sample_count: int,
        point_count: int,
        num_views: int,
        verbose: bool = False,
) -> PointCloud:
    """
    Function to load or create a point cloud (pc) from a given mesh or model path.

    Args:
        mesh_path (Optional[str]): The path to the mesh file. Either this or model_path should be provided.
        model_path (Optional[str]): The path to the model file. Either this or mesh_path should be provided.
        cache_dir (Optional[str]): The directory where cache files are stored. If not provided, caching is not used.
        random_sample_count (int): The number of random points to sample from the multiview.
        point_count (int): The number of points for the final point cloud.
        num_views (int): The number of views to render for creating the multiview.
        verbose (bool, optional): If True, prints out additional details. Defaults to False.

    Returns:
        PointCloud: The created or loaded point cloud.
    """

    # Ensure that either a mesh_path or model_path is provided, but not both
    assert (model_path is not None) ^ (mesh_path is not None), "must specify exactly one of model_path or mesh_path"
    # Decide which path to use
    path = model_path if model_path is not None else mesh_path

    # If a cache_dir is provided
    if cache_dir is not None:
        # Build the cache path
        cache_path = bf.join(
            cache_dir,
            f"pc_{bf.basename(path)}_mat_{num_views}_{random_sample_count}_{point_count}.npz",
        )
        # If the cache file already exists, load it and return
        if bf.exists(cache_path):
            return PointCloud.load(cache_path)
    else:
        cache_path = None

    # If the cache file does not exist, we create a new multiview and point cloud
    with load_or_create_multiview(
            mesh_path=mesh_path,
            model_path=model_path,
            cache_dir=cache_dir,
            num_views=num_views,
            verbose=verbose,
    ) as mv:  # this context manager either loads or creates a multiview
        if verbose:
            print("extracting point cloud from multiview...")
        # Generate a point cloud from the multiview
        pc = mv_to_pc(
            multiview=mv, random_sample_count=random_sample_count, point_count=point_count
        )
        # If a cache_dir is provided, save the point cloud for future use
        if cache_path is not None:
            pc.save(cache_path)
        return pc  # return the point cloud



@contextmanager
def load_or_create_multiview(
    *,
    mesh_path: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    num_views: int = 20,
    extract_material: bool = True,
    light_mode: Optional[str] = None,
    verbose: bool = False,
) -> Iterator[BlenderViewData]:

    assert (model_path is not None) ^ (
        mesh_path is not None
    ), "must specify exactly one of model_path or mesh_path"
    path = model_path if model_path is not None else mesh_path

    if extract_material:
        assert light_mode is None, "light_mode is ignored when extract_material=True"
    else:
        assert light_mode is not None, "must specify light_mode when extract_material=False"

    if cache_dir is not None:
        if extract_material:
            cache_path = bf.join(cache_dir, f"mv_{bf.basename(path)}_mat_{num_views}.zip")
        else:
            cache_path = bf.join(cache_dir, f"mv_{bf.basename(path)}_{light_mode}_{num_views}.zip")
        if bf.exists(cache_path):
            with bf.BlobFile(cache_path, "rb") as f:
                yield BlenderViewData(f)
                return
    else:
        cache_path = None

    common_kwargs = dict(
        fast_mode=True,
        extract_material=extract_material,
        camera_pose="random",
        light_mode=light_mode or "uniform",
        verbose=verbose,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = bf.join(tmp_dir, "out.zip")
        if mesh_path is not None:
            mesh = TriMesh.load(mesh_path)
            render_mesh(
                mesh=mesh,
                output_path=tmp_path,
                num_images=num_views,
                backend="BLENDER_EEVEE",
                **common_kwargs,
            )
        elif model_path is not None:
            render_model(
                model_path,
                output_path=tmp_path,
                num_images=num_views,
                backend="BLENDER_EEVEE",
                **common_kwargs,
            )
        if cache_path is not None:
            bf.copy(tmp_path, cache_path)
        with bf.BlobFile(tmp_path, "rb") as f:
            yield BlenderViewData(f)


def mv_to_pc(multiview: ViewData, random_sample_count: int, point_count: int) -> PointCloud:
    pc = PointCloud.from_rgbd(multiview)

    # Handle empty samples.
    if len(pc.coords) == 0:
        pc = PointCloud(
            coords=np.zeros([1, 3]),
            channels=dict(zip("RGB", np.zeros([3, 1]))),
        )
    while len(pc.coords) < point_count:
        pc = pc.combine(pc)
        # Prevent duplicate points; some models may not like it.
        pc.coords += np.random.normal(size=pc.coords.shape) * 1e-4

    pc = pc.random_sample(random_sample_count)
    pc = pc.farthest_point_sample(point_count, average_neighbors=True)

    return pc


def normalize_input_batch(batch: AttrDict, *, pc_scale: float, color_scale: float) -> AttrDict:
    res = batch.copy()
    scale_vec = torch.tensor([*([pc_scale] * 3), *([color_scale] * 3)], device=batch.points.device)
    res.points = res.points * scale_vec[:, None]

    if "cameras" in res:
        res.cameras = [[cam.scale_scene(pc_scale) for cam in cams] for cams in res.cameras]

    if "depths" in res:
        res.depths = [[depth * pc_scale for depth in depths] for depths in res.depths]

    return res


def process_depth(depth_img: np.ndarray, image_size: int) -> np.ndarray:
    depth_img = center_crop(depth_img)
    depth_img = resize(depth_img, width=image_size, height=image_size)
    return np.squeeze(depth_img)


def process_image(
    img_or_img_arr: Union[Image.Image, np.ndarray], alpha_removal: str, image_size: int
):
    if isinstance(img_or_img_arr, np.ndarray):
        img = Image.fromarray(img_or_img_arr)
        img_arr = img_or_img_arr
    else:
        img = img_or_img_arr
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            # Grayscale
            rgb = Image.new("RGB", img.size)
            rgb.paste(img)
            img = rgb
            img_arr = np.array(img)

    img = center_crop(img)
    alpha = get_alpha(img)
    img = remove_alpha(img, mode=alpha_removal)
    alpha = alpha.resize((image_size,) * 2, resample=Image.BILINEAR)
    img = img.resize((image_size,) * 2, resample=Image.BILINEAR)
    return img, alpha
