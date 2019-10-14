from im2mesh.encoder import conv, pointnet


encoder_dict = {
    'resnet18': conv.Resnet18,
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,
}


encoder_temporal_dict = {
    'conv_3D': conv.ConvEncoder3D,
    'pointnet_2stream': pointnet.ResnetPointnet2Stream,
    'pointnet_resnet': pointnet.TemporalResnetPointnet,
}