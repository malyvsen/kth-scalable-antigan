import torch
import glob
import torchvision
import numpy as np
import torchvision.models as models
from PIL import Image

def get_vector(image):

    t_img = transforms(image)
    my_embedding = torch.zeros(512)

    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())                 

    h = layer.register_forward_hook(copy_data)
    with torch.no_grad():                              
        model(t_img.unsqueeze(0))                      

    h.remove()
    return my_embedding

embedding_list= []
emb_list = []
counter=0
for filename in glob.iglob('images/*.png', recursive=True):
    if counter%1000==0:
        print(counter)
    img = Image.open(filename)
    
    model = models.resnet18(pretrained=True)
    layer = model._modules.get('avgpool')
    model.eval()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pic_vector = get_vector(img)
    pic_vector = np.array(pic_vector) 
    emb_list.append(pic_vector.copy())
    counter=counter+1

np.save('img_embeddings_predefined_resnet_model.npy', emb_list, allow_pickle=True)







