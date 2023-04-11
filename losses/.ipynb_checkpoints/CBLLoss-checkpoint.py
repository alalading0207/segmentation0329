import torch
import torch.nn.functional as F


# cbl_loss = CBLLoss([3,3], 10)
# cbl_loss(cbl_1_8, label_1_8)
class CBLLoss(torch.nn.Module):
    def __init__(self, kernel_size, tau, use_kl=False):
        super().__init__()
        self.tau = tau
        self.use_kl = use_kl
        self.epslion = 1e-8
        # [h, w]
        self.kernel_height = kernel_size[0]
        self.kernel_width = kernel_size[1]
        self.neibor_size = self.kernel_height * self.kernel_width - 1
        self.padding_size = int((self.kernel_height - 1)/2)
        print('padding_size:',self.padding_size)
    

    def forward(self, features, boundry):
        # features = features/(torch.norm(features,dim=1,keepdim=True))
        return self.cal_dist(features, boundry)

    def cal_dist(self, features, boundry):
        n, c, h, w = boundry.shape   # bs,1,32,32
        device = boundry.device

        # 创建邻域张量
        distance = torch.zeros([n, self.neibor_size, h, w], device=device)   # feature上  bs,8,32,32
        label = torch.zeros([n, self.neibor_size, h, w], device=device)      # boundary上 bs,8,32,32

        # padding 
        padded_features = F.pad(features, [self.padding_size, self.padding_size, self.padding_size, self.padding_size])   # bs,512,34,34
        # bs,1,34,34  边缘一圈是-1      boundry+1的值有1, 2       padded_boundry的值有-1, 1, 2     
        padded_boundry = F.pad(boundry+1, [self.padding_size, self.padding_size, self.padding_size, self.padding_size], mode='constant', value=-1)


        for i in range(self.kernel_height):   # i:  0,1,2           j:  0,1,2           index:  0,1,2,  3,4,5,   6,7,8
            for j in range(self.kernel_width):
                index = i*self.kernel_width+j

                if index==(self.neibor_size/2) :   # 中心点
                    pass

                elif index > (self.neibor_size/2) :
                    if self.use_kl:
                        distance[:, index-1:index, :,:] = self.dist_kl(padded_features[:, :, i:i+h, j:j+w], features)
                    else:    
                        distance[:, index-1:index, :,:] = self.dist_l2(padded_features[:, :, i:i+h, j:j+w], features)  

                    label[:, index-1:index, :, :] = padded_boundry[:, :, i:i+h, j:j+w] * (boundry+1)      

                else:
                    if self.use_kl:
                        distance[:, index:index+1, :,:] = self.dist_kl(padded_features[:, :, i:i+h, j:j+w], features)
                    else:    
                        # features中每一个点与其8邻域的距离存储再distance的ch维中
                        distance[:, index:index+1, :,:] = self.dist_l2(padded_features[:, :, i:i+h, j:j+w], features)  
                    # -1,-2则为边界
                    label[:, index:index+1, :, :] = padded_boundry[:, :, i:i+h, j:j+w] * (boundry+1)   

        

        valid_point = label > 0         # bs,8,32,32
        same_bou_point =  (label==4)    # i点与邻域j点 同为边界点

        num_bou_points = torch.sum(boundry)    # 边界点个数
        valid_bou_points =  torch.sum(torch.any(same_bou_point, dim=1))   # 有效边界点个数: 邻域有边界点的 中心边界点的个数

        distance = torch.exp(-distance / self.tau)
        numerator = torch.sum(distance * same_bou_point.float(), dim=1)    # 分子是0代表：1.该点不是边界点 2.该点是边界点 但邻域里没有边界点
        denominator = torch.sum(distance*valid_point.float(), dim=1)       # bs,32,32 

        every_point_log = torch.log(numerator / (denominator + self.epslion))
        every_point_log_ = torch.where(torch.isinf(every_point_log), torch.full_like(every_point_log, 0), every_point_log)
        # every_point_log_ = torch.where(torch.isinf(every_point_log), torch.zeros_like(every_point_log), every_point_log)

        '''问题: 跑一个batch后卷积权重变成了nan'''

        return (-1 / valid_bou_points) * (torch.sum(every_point_log_))
    


    
    def dist_kl(self, neighbours, features):
        log_p = features.softmax(dim=1)
        log_q = neighbours.softmax(dim=1)
        return torch.sum(log_p*(log_p.log()-log_q.log()), dim=1, keepdim=True)

    def dist_l2(self, neighbours, features):
        return torch.norm(neighbours-features, p=2, dim=1, keepdim=True)   # 在ch上求2范数