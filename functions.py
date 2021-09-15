max_ent = np.zeros(100000, dtype=float).reshape(10,10000) # matrix which stores values for each image of entropy
margin = np.zeros(100000, dtype=float).reshape(10,10000)# matrix which stores values for each image of difference between highest and second highest probability

def maximum_entropy(output,image_index,batch_size):
    log_output = torch.log(output)
    mul = output*log_output
    result = torch.sum(mul, dim=0)
    image_index = image_index.cpu().numpy()
    for batch_number in range(batch_size):
        ind = image_index[batch_number] # index of the image 
        max_ent[ind] = result[batch_number].detach().cpu().numpy()  # storing entropy to matrix

def margin(output, image_index, batch_size):
    sorting, indices = torch.sort(output, descending=True)    
    image_index = image_index.cpu().numpy()
    for batch_number in range(batch_size): 
        margin[ind] = sorting[0][batch_number].detach().cpu().numpy()- sorting[1][batch_number].detach().cpu().numpy()  
        



       
