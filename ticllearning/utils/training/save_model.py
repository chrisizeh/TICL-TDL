import os
import torch
import copy

def save_model(model, epoch, optimizer, loss, val_loss, output_folder, filename, dummy_input=None):
    path = os.path.join(output_folder, f"{filename}")

    print(f">>> Saving model to {path}")
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': loss,
                'validation_loss': val_loss
                }, f"{path}_epoch_{epoch}_dict.pt")
    
    if (dummy_input is not None):
        # Move model to cpu before tracing
        dump_model = copy.deepcopy(model)
        dump_model.to("cpu")

        # Double-check any buffers/constants
        for name, buf in dump_model.named_buffers(recurse=True):
            if buf.device.type != "cpu":
                dump_model.register_buffer(name, buf.cpu(), persistent=True)

        for name, param in dump_model.named_parameters(recurse=True):
            if param.device.type != "cpu":
                param.data = param.cpu()
        dump_model.eval()
        
        with torch.no_grad():
            dummy_input_copy = copy.deepcopy(dummy_input)
            dummy_input_copy.to("cpu")
            test_input = (dummy_input_copy.x, dummy_input_copy.edge_features, dummy_input_copy.edge_index)
            traced_model = torch.jit.script(dump_model)
            
            if (torch.allclose(dump_model(*test_input), traced_model(*test_input), atol=1e-4)):
                traced_model.save(f"{path}_traced.pt")
            else:
                traced_model.save(f"{path}_diff_traced.pt")
                print("Traced model is not similar to python model.")
    else:
        torch.save(model, f"{path}_pickle.pt")
    model.train()


