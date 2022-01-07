
import numpy as np
import time

def model_train(epoch, train_dataloader, device, reconstructor, criterion, optimizer):

    total_loss = 0
    accuracy = []
    start_time = time.time()
    
    for i, batch in enumerate(train_dataloader, 0):
        print(batch)
        images = batch[0].to(device)
        labels = batch[1].to(device)

        output = reconstructor(images) 
        
        loss = criterion(output, labels) 
        total_loss += loss.item()

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()
        
        argmax = output.argmax(dim=1) 
        accuracy.append((labels==argmax).sum().item() / labels.shape[0]) 

        print('Epoch: [{}]/({}/{}), Train Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
            epoch, i, len(train_dataloader), loss.item(), sum(accuracy)/len(accuracy), time.time()-start_time ))
    
    return total_loss / len(train_dataloader) 
