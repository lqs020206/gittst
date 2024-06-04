import torch

def map_labels_to_positions(label1):
    positions = {
        0: 0,
        1000: 1,
        1100: 2,
        1110: 3,
    }
    positions_list = []
    label1 = torch.round(label1)
    for label in label1:
        label_int = int(torch.sum(label * torch.tensor([1000, 100, 10, 1])))
        position = positions[label_int]
        positions_list.append(position)
    return torch.tensor(positions_list).unsqueeze(1)

# 测试
label1 = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]])
positions_mapped = map_labels_to_positions(label1)
print(positions_mapped)