import torch
import torch.nn.functional as F

class CBLLoss(torch.nn.Module):
    def __init__(self, kernel_size, tau, use_kl=False):
        super().__init__()
        self.tau = tau
        self.use_kl = use_kl
        # [h, w]
        self.kernel_height = kernel_size[0]
        self.kernel_width = kernel_size[1]
        self.padding_size = int((self.kernel_height - 1)/2)
        print(self.padding_size)
    

    def forward(self, features, boundry):
        return self.cal_dist(features, boundry)

    def cal_dist(self, features, boundry):
        n, c, h, w = boundry.shape
        device = boundry.device
        distance = torch.zeros([n, self.kernel_height * self.kernel_width, h, w], device=device)
        # <0: nothing to compare
        # 1: same
        # 2: different
        label = torch.zeros([n, self.kernel_height * self.kernel_width, h, w], device=device)
        # padding 
        padded_features = F.pad(features, [self.padding_size, self.padding_size, self.padding_size, self.padding_size])
        padded_boundry = F.pad(boundry+1, [self.padding_size, self.padding_size, self.padding_size, self.padding_size], mode='constant', value=-1) 
        for i in range(self.kernel_height):
            for j in range(self.kernel_width):
                index = i*self.kernel_width+j
                if self.use_kl:
                    distance[:, index:index+1, :,:] = self.dist_kl(padded_features[:, :, i:i+h, j:j+w], features)
                else:
                    distance[:, index:index+1, :,:] = self.dist_l2(padded_features[:, :, i:i+h, j:j+w], features)
                label[:, index:index+1, :, :] = padded_boundry[:, :, i:i+h, j:j+w] * (boundry+1)
        valid_points = label > 0
        same_label_points = (label==1) + (label==4)
        distance = torch.exp(-distance / self.tau)
        numerator = torch.sum(distance * same_label_points.float(), dim=1) 
        denominator = torch.sum(distance*valid_points.float(), dim=1)
        return (-1/(h*w)) * (torch.sum(torch.log(numerator / denominator), dim=[1,2]))

    
    def dist_kl(self, neighbours, features):
        log_p = features.softmax(dim=1)
        log_q = neighbours.softmax(dim=1)
        return torch.sum(log_p*(log_p.log()-log_q.log()), dim=1, keepdim=True)

    def dist_l2(self, neighbours, features):
        return torch.norm(neighbours-features, p=2, dim=1, keepdim=True)