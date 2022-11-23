require "numo/narray"
require "torch"
require "torchvision"
require "tensorflow"

# transforms
transform = TorchVision::Transforms::Compose.new([
  TorchVision::Transforms::ToTensor.new,
  TorchVision::Transforms::Normalize.new([0.5], [0.5])
])

# datasets
trainset = TorchVision::Datasets::FashionMNIST.new("./data", train: true, download: true, transform: transform)
testset = TorchVision::Datasets::FashionMNIST.new("./data", train: false, download: true, transform: transform)

# dataloaders
trainloader = Torch::Utils::Data::DataLoader(trainset, batch_size: 4, shuffle: true, num_workers: 2)


testloader = Torch::Utils::Data::DataLoader(testset, batch_size: 4, shuffle: false, num_workers: 2)

# constant for classes
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False)
  img = img.mean(dim=0) if one_channel
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  if one_channel
    plt.imshow(npimg, cmap="Greys")
  else
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
  end
end

class Net < Torch::NN::Module
  def initialize
    super()
    @conv1 = Torch::NN::Conv2d.new(1, 6, 5)
    @conv2 = Torch::NN::Conv2d.new(6, 16, 5)
    @fc1 = Torch::NN::Linear.new(16 * 4 * 4, 120)
    @fc2 = Torch::NN::Linear.new(120, 84)
    @fc3 = Torch::NN::Linear.new(84, 10)
  end

  def forward(x)
    x = Torch::NN::F.max_pool2d(Torch::NN::F.relu(@conv1.call(x)), [2, 2])
    x = Torch::NN::F.max_pool2d(Torch::NN::F.relu(@conv2.call(x)), 2)
    x = Torch.flatten(x, 1)
    x = Torch::NN::F.relu(@fc1.call(x))
    x = Torch::NN::F.relu(@fc2.call(x))
    @fc3.call(x)
  end
end


net = Net.new
criterion = Torch::NN::CrossEntropyLoss()
optimizer = Torch::Optim::SGD.new(net.parameters, lr: 0.001, momentum: 0.9)


writer = Tensorflow::Utils::Tensorboard::SummaryWriter('runs/fashion_mnist_experiment_1')


