import argparse
import yaml
from nerf_helper import *
import os 
import cv2
from skimage.metrics import structural_similarity as ssim 
from skimage import io

def get_parser():
    #configs:
    parser = argparse.ArgumentParser(description='Nia-39-1 NeRF Tutorial')
    parser.add_argument('--resized_res',type=int, default=5)
    parser.add_argument('--config',default='./config.yaml',required=True)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--rendering', type=bool, default=False) 
    parser.add_argument('--gpu_num', type=str, default=0,required=True)     
    return parser

def get_data(tgt_class,resized_res):
    
    print('...get data (tgt_class) : {}'.format(tgt_class))
    
    data = load_data(tgt_class,json_path='./dataset/transforms.json',resized_res=resized_res)
    
    images_arr = data['images'][:,:,:,:3]
    poses_arr = data['poses']
    focal_arr = data['focal']
    
    height, width = images_arr.shape[1:3]
    
    near, far = 2., 6.
    
    images = torch.from_numpy(images_arr).to(device)
    poses = torch.from_numpy(poses_arr).to(device)
    focal = torch.from_numpy(focal_arr).to(device)
    
    torch_data = {'height':height,'width':width, 'near':near,'far':far
                  ,'images':images,'poses':poses,'focal':focal}
    
    return torch_data

def init_models():
        
    # Encoders
    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    encode = lambda x: encoder(x)

    # View direction encoders
    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,log_space=log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())
    if use_fine_model:
        fine_model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
                          d_viewdirs=d_viewdirs)
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=np.float32(lr))

    # Early Stopping
    warmup_stopper = EarlyStopping(patience=50)

    return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper

def nerf_forward(rays_o,rays_d,near,far,encoding_fn,coarse_model,kwargs_sample_stratified = None,n_samples_hierarchical = 0
                 ,kwargs_sample_hierarchical = None,fine_model = None,viewdirs_encoding_fn = None,chunksize = 2**15):
    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
      rays_o, rays_d, near, far, **kwargs_sample_stratified)

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                   viewdirs_encoding_fn,
                                                   chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
    
    outputs = {'z_vals_stratified': z_vals}

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical(rays_o, rays_d, z_vals, weights
                                                                        ,n_samples_hierarchical
                                                                        ,**kwargs_sample_hierarchical)

        # Prepare inputs as before.
        batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                     viewdirs_encoding_fn,
                                                     chunksize=chunksize)
        else:
            batches_viewdirs = [None] * len(batches)

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
            
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)

        # Store outputs.
        outputs['z_vals_hierarchical'] = z_hierarch
        outputs['rgb_map_0'] = rgb_map_0
        outputs['depth_map_0'] = depth_map_0
        outputs['acc_map_0'] = acc_map_0

    # Store outputs.
    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    return outputs

def train():
    
    for key,value in params.items():
        globals()[key] =value
    # Shuffle rays across all images.
    
    for key,value in data.items():
        globals()[key] =value
        
    if not one_image_per_step:
        height, width = images.shape[1:3]
        all_rays = torch.stack([torch.stack(get_rays(height, width, focal, p), 0)
                               for p in poses[:trn_idx]], 0)
        rays_rgb = torch.cat([all_rays, images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []
    
    for i in trange(n_iters):
        model.train()

        if one_image_per_step:
            # Randomly pick an image as the target.
            trn_idxes = np.arange(trn_idx)
            np.random.shuffle(trn_idxes)
            target_img_idx = trn_idxes[0]
            target_img = images[target_img_idx].to(device)
            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = poses[target_img_idx].to(device)
            rays_o, rays_d = get_rays(height, width, focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            
        else:
            # Random over all images.
            batch = rays_rgb[i_batch:i_batch + batch_size]
            batch = torch.transpose(batch, 0, 1)
            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += batch_size
            
            # Shuffle after one epoch
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        
        target_img = target_img.reshape([-1, 3])

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        outputs = nerf_forward(rays_o, rays_d,
                               near, far, encode, model,
                               kwargs_sample_stratified=kwargs_sample_stratified,
                               n_samples_hierarchical=n_samples_hierarchical,
                               kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                               fine_model=fine_model,
                               viewdirs_encoding_fn=encode_viewdirs,
                               chunksize=chunksize)

        # Check for any numerical issues.
        for k, v in outputs.items():
            
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        # Backprop!
        rgb_predicted = outputs['rgb_map']

        loss = torch.nn.functional.mse_loss(rgb_predicted.float(), target_img.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute mean-squared error between predicted and target images.
        psnr = -10. * torch.log10(loss)
        train_psnrs.append(psnr.item())

        # Evaluate testimg at given display rate.
        if i % display_rate == 0:
            model.eval()
            height, width = images[tst_idx].shape[:2]
            
            rays_o, rays_d = get_rays(height, width, focal, poses[tst_idx])
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            outputs = nerf_forward(rays_o, rays_d,
                                 near, far, encode, model,
                                 kwargs_sample_stratified=kwargs_sample_stratified,
                                 n_samples_hierarchical=n_samples_hierarchical,
                                 kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                                 fine_model=fine_model,
                                 viewdirs_encoding_fn=encode_viewdirs,
                                 chunksize=chunksize)

            rgb_predicted, depth_predicted = outputs['rgb_map'], outputs['depth_map']
            loss = torch.nn.functional.mse_loss(rgb_predicted, images[tst_idx].reshape(-1, 3))

            val_psnr = -10. * torch.log10(loss)

            val_psnrs.append(val_psnr.item())
            print("Loss(mse):", loss.item(), "train_psnr:", psnr.item(), "val_psnr:", val_psnr.item())
            iternums.append(i)

            # Plot example outputs   
            fig, ax = plt.subplots(1, 4, figsize=(24,4), gridspec_kw={'width_ratios': [1, 1, 1, 2]})

            ax[0].imshow(images[tst_idx].detach().cpu().numpy())
            ax[0].set_title(f'iter. : {i} | Target')

            ax[1].imshow(rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
            ax[1].set_title(f'predicted rgb')

            ax[2].imshow(depth_predicted.reshape([height,width]).detach().cpu().numpy())
            ax[2].set_title('predicted depth')


            ax[3].plot(range(0, i + 1), train_psnrs, 'r',label='train')
            ax[3].plot(iternums, val_psnrs, 'b',label='valid')
            ax[3].legend()

            ax[3].set_title('PSNR {:.2}'.format(val_psnrs[-1]))

            plt.savefig(resultdir+f'result_{i}.png')
            plt.close()


        # Check PSNR for issues and stop if any are found.
        if i == warmup_iters - 1:
            if val_psnr < warmup_min_fitness:
                print(f'Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...')
                return False, train_psnrs, val_psnrs
            
        elif i < warmup_iters:
            if warmup_stopper is not None and warmup_stopper(i, psnr):
                print(f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...')
                return False, train_psnrs, val_psnrs

    return True, train_psnrs, val_psnrs

def pose_spherical(theta, phi, radius): # camera2world
    trans_t = lambda t : torch.Tensor([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,t],
                                   [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([[1,0,0,0],
                                         [0,np.cos(phi),-np.sin(phi),0],
                                         [0,np.sin(phi), np.cos(phi),0],
                                         [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([[np.cos(th),0,-np.sin(th),0]
                                         ,[0,1,0,0]
                                         ,[np.sin(th),0, np.cos(th),0]
                                         ,[0,0,0,1]]).float()

    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def rendering(tgt_class):
    _, _, encode, encode_viewdirs, optimizer, warmup_stopper = init_models() 
    ckpt = torch.load('./logs/{}_model.tar'.format(tgt_class))
    model, fine_model = ckpt['model'], ckpt['fine_model']

    frames = []
    poses = [] 

    for th in np.linspace(0., 360., 100, endpoint=False):
        c2w = pose_spherical(th, -20., 3.5).to("cuda") 
        poses.append(c2w.detach().cpu().numpy())
        rays_o, rays_d = get_rays(height, width, focal, c2w)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])

        outputs = nerf_forward(rays_o, rays_d,
                             near, far, encode, model,
                             kwargs_sample_stratified=kwargs_sample_stratified,
                             n_samples_hierarchical=n_samples_hierarchical,
                             kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                             fine_model=fine_model,
                             viewdirs_encoding_fn=encode_viewdirs,
                             chunksize=2**15)
        rgb = outputs['rgb_map']
        rgb = rgb.reshape([height, width, 3]).detach().cpu().numpy()
        img = (255*np.clip(rgb,0,1)).astype(np.uint8)
        frames.append((255*np.clip(rgb,0,1)).astype(np.uint8))

    f = renderingdir+'{}_video.gif'.format(tgt_class)
    imageio.mimwrite(f, frames, fps=30)

def mtjin_bgr2gray(bgr_img):
    # BGR 색상값
    b = bgr_img[:, :, 0]*255
    g = bgr_img[:, :, 1]*255
    r = bgr_img[:, :, 2]*255
    result = ((0.299 * r) + (0.587 * g) + (0.114 * b))
    # imshow 는 CV_8UC3 이나 CV_8UC1 형식을 위한 함수이므로 타입변환
    return result.astype(np.uint8)

def testing(tgt_class):
    psnrs = []
    ssims = []
    _, _, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()
    ckpt = torch.load('./logs/{}_model.tar'.format(tgt_class))
    model, fine_model = ckpt['model'], ckpt['fine_model']

    for idx,ds_name in zip(range(19,21),['ValSet','TestSet']):

        pose = poses[idx]
        rays_o, rays_d = get_rays(height, width, focal, pose)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])

        outputs = nerf_forward(rays_o, rays_d,
                             near, far, encode, model,
                             kwargs_sample_stratified=kwargs_sample_stratified,
                             n_samples_hierarchical=n_samples_hierarchical,
                             kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                             fine_model=fine_model,
                             viewdirs_encoding_fn=encode_viewdirs,
                             chunksize=2**15)

        imageA = outputs['rgb_map']#.astype(np.uint8)
        imageA = imageA.reshape([height, width, 3]).detach().cpu().numpy()
        imageB = images[idx]#.astype(np.uint8)
        imageB = imageB.detach().cpu().numpy()
        imageB = np.array(imageB, dtype=np.float32)

        plt.suptitle(ds_name,y=0.85)
        plt.subplot(1,2,1)
        plt.title('pred_rgb')
        plt.imshow(imageA)
        plt.subplot(1,2,2)
        plt.title('true_rgb')
        plt.imshow(imageB)
        plt.savefig(testdir+'{}_result.png'.format(ds_name))
        plt.close()

        # 4. Convert the images to grayscale
        grayA =  mtjin_bgr2gray(imageA)
        grayB =  mtjin_bgr2gray(imageB)

        # 5. Compute the Structural Similarity Index (SSIM) between the two
        #    images, ensuring that the difference image is returned
        (score, diff) = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        ssims.append(score)


        imageA = outputs['rgb_map']#.astype(np.uint8)
        imageA = imageA.reshape([height, width, 3])
        imageB = images[idx]#.astype(np.uint8)

        loss = torch.nn.functional.mse_loss(imageA.reshape(-1, 3), imageB.reshape(-1, 3))
        psnr = -10. * torch.log10(loss)
        psnrs.append(psnr.item())

        print('ds_name : {} psnr : {:.4} ssim : {:.4}'.format(ds_name, psnr,score))

if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()
    
    # define y = x 
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    params = config['parameters']
       
    for key,value in params.items():
        globals()[key] = value
        
    # GPU Setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For repeatability
    seed = 3407
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('1. Device:', device)
    print('2. Current cuda device:', torch.cuda.current_device())
    print('3. Count of using GPUs:', torch.cuda.device_count())
    print('4. cuda available:', torch.cuda.is_available())
    print('5. torch version:', torch.__version__)

    data = get_data(tgt_class,resized_res)    
    
    for key,value in data.items():
        globals()[key] =value
           
    resultdir = savedir+f'checkpoint/{tgt_class}/'
    renderingdir = savedir+f'rendering/'
    testdir = savedir+f'testing/{tgt_class}/'
    
    for folder in [savedir, resultdir, renderingdir,testdir]:
        os.makedirs(folder,exist_ok=True)

    if args.training:
        for _ in range(10000):
            start = time.time()
            model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()
            success, train_psnrs, val_psnrs = train()

            if success and val_psnrs[-1] >= warmup_min_fitness:
                
                print('Training successful!')
                path = savedir+f'/model/{tgt_class}_model.tar'
                torch.save({'model':model,'fine_model':fine_model}, path)
                print('Saved checkpoints at', path)
                end = time.time()

                cost_time = (end - start) 
                h,m,s = cost_time//(60*60),(cost_time%(60*60))//60, (cost_time%(60*60))%60 
                print('{:.4} hour {:.4} min {:.4} sec'.format(h,m,s))
                break
        
        print('')
        print(f'Training Done!')

    if args.testing:
        testing(tgt_class)        
        print('')
        print(f'Testing Done!') 
              
    if args.rendering:
        rendering(tgt_class)
        print('')
        print(f'Rendering Done!') 
               
