import os
import torch
import tqdm


with open('/home/jovyan/Tatiana_Z/ScanNet/split/scannet_train_scans.txt', "r") as f:
    scene_ids = f.readlines()

features_dict = {}
for scene in tqdm.tqdm(scene_ids):
    scene_file = os.path.join("/home/jovyan/Tatiana_Z/output_vlsat/", scene.strip()+".pt")
    try:
        features_dict[scene] = []
        scene_edge_feat = torch.load(scene_file)
        for hash_key, hash_value in scene_edge_feat.items():
            #edge_feat = torch.load(os.path.join(scene_dir, filename))
            # 'windowsill_21_supported by_washing powder_18.pt'
            label_1, id_1, _, label_2, id_2 = hash_key.split("_")
            features_dict[f"{scene}_{id_1}_{id_2}"] = hash_value
    except:
        print("EXCLUDE", scene)

torch.save(features_dict, f"/home/jovyan/Tatiana_Z/Chat-3D-v2/annotations/scannet_train_mask3d_gnn_feats.pt")
