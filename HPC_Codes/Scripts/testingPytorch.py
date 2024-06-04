
import torch

def main():
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU for computations.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU for computations.")

    # Create a random tensor and move it to the selected device
    x = torch.rand(5, 3).to(device)
    print("Random tensor:")
    print(x)

    # Perform a simple operation on the tensor
    y = torch.rand(5, 3).to(device)
    result = torch.matmul(x, y.t())  # Matrix multiplication
    print("Result of matrix multiplication:")
    print(result)

if __name__ == "__main__":
    main()
