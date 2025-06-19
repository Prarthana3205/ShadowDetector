from torch.utils.data import DataLoader
from shadow_dataset import ShadowScoutDataset

dataset = ShadowScoutDataset("preprocessed")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    print(batch.shape)
    break
