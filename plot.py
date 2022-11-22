import numpy as np
import os
mode="fid"
if mode=="fid":
    if os.path.isdir("./checkpoints/cityscapes_tile/MIOU"):
        miou_array = np.load("./checkpoints/oasis_cityscapes_wavelet_disc/MIOU/miou_log.npy")
        print("len", miou_array.shape)
        miou_max = np.amax(miou_array[1,:])
        max_index = np.argmax(miou_array[1,:])
        print("max", np.amax(miou_array[1,:]))
        print("max_index", max_index)
        max_iter = miou_array[0, max_index]
        print("max_iter", max_iter)

        fid_array = np.load("./checkpoints/oasis_cityscapes_wavelet_disc/FID/fid_log.npy")
        print("len", fid_array.shape)
        fid_max = np.amin(fid_array[1,:])
        max_index = np.argmin(fid_array[1,:])
        print("fid", np.amin(fid_array[1,:]))
        print("fid_index", max_index)
        max_iter = fid_array[0, max_index]
        print("fid_iter", max_iter)
        print("miou at best fid", miou_array[1, max_index])
    else:
        fid_array = np.load("./checkpoints/oasis_cityscapes_wavelet_disc/FID/fid_log.npy")
        print("len", fid_array.shape)
        fid_max = np.amin(fid_array[1, :])
        max_index = np.argmin(fid_array[1, :])
        print("fid", np.amin(fid_array[1, :]))
        print("fid_index", max_index)
        max_iter = fid_array[0, max_index]
        print("fid_iter", max_iter)
        print("-----------------------------------------------------------------------------------------------------------------")
        for row in np.transpose(fid_array):
            print("{:d} {:.3f} \\\\".format(int(row[0]), row[1]))
else:
    loss_array = np.load("./checkpoints/oasis_cityscapes_no_edge/losses/losses.npy", allow_pickle=True).item()
    loss_array_gan =  np.asarray(loss_array["GAN"])
    n = np.array(range(len(loss_array_gan))) * 250
    print(len(n))
    with open("./log_loss_.txt", 'w') as out_file:
        for i in range(len(n)):
            out_file.write("{:d} {:.3f} \\\\".format(int(n[i]), loss_array_gan[i]))





