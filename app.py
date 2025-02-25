import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from kan import KAN, plot_KAN
import theory 


css = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap');

.gradio-container {
    width: 1232px !important;
    min-width: 1232px !important;
    margin: 0 auto !important;
}

#interface-container {
    max-width: 1202px; 
    min-width: 1202px; 
    margin: 0 auto;  
    width: 1202px;
}

main.fillable.svelte-d1xpwf.app {
    width: 1202px !important;
    margin: 0 15px !important;
}

:root {
    --line-lg: 1.4;
    --size-8: 0;
}

#shadow_block {
    padding: 12px;
    box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
}

#theory_container{
    width: 805px !important;
    margin: 50px auto !important;
}
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dictionaries and lists for selecting parameters
RESIDUAL_FN = {"SiLU": nn.SiLU, "ReLU": nn.ReLU, "ELU": nn.ELU, "Mish": nn.Mish}

OPTIMIZERS = {"Adam": optim.Adam}

# Mapping function names to (lambda function, input dimension)
DATA_FN = {
    "exp((sin(pi*x)+y^2)/2)-1": (lambda X: torch.exp((torch.sin(torch.pi * X[:, 0]) + X[:, 1]**2)/2)-1, 2),    
    "(2x+3)/(2y+3)-1": (lambda X: (2 * X[:, 0] + 3) / (2 * X[:, 1] + 3) - 1, 2),
    "sqrt((x^2+y^2+z^2)/3)": (lambda X: torch.sqrt((X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2) / 3), 3),
    "tanh(x^4-y^4)": (lambda X: torch.tanh(X[:, 0]**4 - X[:, 1]**4), 2),
    "xy": (lambda X: X[:, 0] * X[:, 1], 2),
    "arcsin(x*sin(y))": (lambda X: torch.arcsin(X[:, 0] * torch.sin(X[:, 1])), 2),
}

RESIDUAL_FN_LIST = list(RESIDUAL_FN.keys())
OPTIMIZERS_LIST = list(OPTIMIZERS.keys())
DATA_FN_LIST = list(DATA_FN.keys())


def generate_data(data_fn, n_samples, noise=0.1, seed=42):
    torch.manual_seed(seed)
    f, in_dim = DATA_FN[data_fn]
    X = torch.rand(n_samples, in_dim) * 2 - 1
    target = f(X)
    target += torch.normal(mean=torch.zeros_like(target), std=noise)
    return X, target.view(-1, 1)


def generate_dataset(data_fn, train_test_ratio=0.5, noise=0.1, batch_size=100, seed=42):
    torch.manual_seed(seed)
    n_train_samples = int(2000 * train_test_ratio)
    train_data, train_labels = generate_data(data_fn, n_train_samples, noise, seed)
    n_test_samples = 2000 - n_train_samples
    test_data, test_labels = generate_data(data_fn, n_test_samples, noise, seed + 100)
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    
    return {
        "test_data": test_data.to(device),
        "test_labels": test_labels.to(device),
        "train_loader": train_loader
    }


def update_data(data_fn, train_test_ratio, noise, batch_size, state, seed):
    state["status"] = "stop"
    state["in_dim"] = DATA_FN[data_fn][1]
    state["seed"] = seed
    state["epoch"] = 0
    state["data"] = generate_dataset(data_fn, train_test_ratio, noise, batch_size, seed)
    return state


def update_model_parameters(layers, grid_size, spline_order, residual_fn, state):
    state["layers"] = layers
    state["grid_size"] = grid_size
    state["spline_order"] = spline_order
    state["residual_fn"] = residual_fn
    return state


def update_learning_parameters(optimizer, lr, reg_activation, reg_entropy, state):
    state["optimizer"] = optimizer
    state["lr"] = float(lr)
    if state["epoch"] > 0:
        state["change_lr"] = True
    state["reg_activation"] = float(reg_activation)
    state["reg_entropy"] = float(reg_entropy)
    return state


def run_pause_train(state):
    if state["status"] == "run":
        state["status"] = "pause"
        return "RUN"
    else:
        state["status"] = "run"
        return "PAUSE"


def stop_train(state):
    state["status"] = "stop"
    state["epoch"] = 0
    state["change_lr"] = False
    return "RUN", "0"


def train(state):
    model = state["model"]
    data = state["data"]
    criterion = nn.MSELoss().to(device)
    optimizer = OPTIMIZERS[state["optimizer"]](model.parameters(), lr=state["lr"])
    if state["epoch"] > 0:
        optimizer.load_state_dict(state["optimizer_state"])
    
    while state["status"] == "run":        
        if state["change_lr"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = state["lr"]
            state["change_lr"] = False
        
        model.train()
        total_train_loss = 0.0
        for batch_data, batch_labels in data["train_loader"]:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            mse_loss = criterion(outputs, batch_labels)
            reg_loss = model.regularization_loss(state["reg_activation"], state["reg_entropy"])
            train_loss = mse_loss + reg_loss
            train_loss.backward()
            optimizer.step()
            total_train_loss += mse_loss.item()

        model.eval()
        with torch.no_grad():
            test_outputs = model(data["test_data"])
            test_rmse = float(torch.sqrt(criterion(test_outputs, data["test_labels"])))
        train_rmse = float(torch.sqrt(torch.tensor(total_train_loss / len(data["train_loader"]))))              
        new_loss = pd.DataFrame({
            "data": ["train", "test"],
            "x": [int(state["epoch"]) + 1, int(state["epoch"]) + 1],
            "y": [train_rmse, test_rmse]
        })

        if state["status"] == "run":
            state["loss_df"] = pd.concat([state["loss_df"], new_loss], ignore_index=True)
            state["model"] = model
            state["optimizer_state"] = optimizer.state_dict()
            state["epoch"] += 1
            yield (
                plot_KAN(model),                        # kan_plot
                str(state["epoch"]),                    # epoch
                f" {test_rmse:.5f}<br />{train_rmse:.5f}<br />{float(reg_loss):.5f}", # losses
                state["loss_df"]                        # loss_plot
            )


def update_model(state):
    torch.manual_seed(state["seed"])
    data = state["data"]
    model = KAN([state["in_dim"], *state["layers"], 1],
                grid_size=state["grid_size"], spline_order=state["spline_order"],
                residual_fn=RESIDUAL_FN[state["residual_fn"]]).to(device)
    criterion = nn.MSELoss().to(device)

    model.eval()
    with torch.no_grad():
        total_train_loss = 0.0
        for batch_data, batch_labels in data["train_loader"]:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            total_train_loss += criterion(outputs, batch_labels).item()
            reg_loss = model.regularization_loss(state["reg_activation"], state["reg_entropy"])
        train_rmse = float(torch.sqrt(torch.tensor(total_train_loss / len(data["train_loader"]))))
        test_outputs = model(data["test_data"])
        test_rmse = float(torch.sqrt(
            criterion(test_outputs, data["test_labels"])))
    
    state["loss_df"] = pd.DataFrame({
        "data": ["train", "test"],
        "x": [0, 0],
        "y": [float(train_rmse), float(test_rmse)]
    })
    state["model"] = model

    # Compute the total number of trainable parameters
    num_params = state["in_dim"] * state["layers"][0] + state["layers"][-1]
    for i in range(len(state["layers"]) - 1):
        num_params += state["layers"][i] * state["layers"][i + 1]
    num_params *= state["grid_size"] + state["spline_order"] + 1

    return (
        plot_KAN(model),                                # kan_plot
        str(state["epoch"]),                            # epoch
        f"{test_rmse:.5f}<br />{train_rmse:.5f}<br />{float(reg_loss):.5f}", # losses
        state["loss_df"],                               # loss_plot
        "Total number of parameters " + str(num_params) # num_params
    )


def num_layers_change(num_layers, layers):
    if num_layers > len(layers):
        return layers + [1]
    else:
        return layers[:-1]


INIT_CONFIG = {
    "data_fn": DATA_FN_LIST[0],
    "layers": [2, 1],
    "grid_size": 5,
    "spline_order": 3,
    "residual_fn": RESIDUAL_FN_LIST[0],
    "train_test_ratio": 0.7,
    "noise": 0.03,
    "batch_size": 50,
    "optimizer": OPTIMIZERS_LIST[0],
    "lr": 0.01,
    "reg_activation": 0.01,
    "reg_entropy": 0.01,
    "seed": 42
}


with gr.Blocks(css=css) as demo:

    state = gr.State({
        "status": "stop", # values: "run", "pause", "stop"
        "epoch": 0,
        "loss_df": None, # DataFrame for loss_plot
        "in_dim": DATA_FN[INIT_CONFIG["data_fn"]][1],
        "layers": INIT_CONFIG["layers"], #hidden layers dims
        "grid_size": INIT_CONFIG["grid_size"],
        "spline_order": INIT_CONFIG["spline_order"],
        "residual_fn": INIT_CONFIG["residual_fn"],
        "model": None,
        "data": None, # dict with keys "test_data", "test_labels", "train_loader"
        "lr": INIT_CONFIG["lr"],
        "change_lr": False, # trigger for updating lr in training
        "optimizer": INIT_CONFIG["optimizer"],
        "optimizer_state": None,
        "reg_activation": INIT_CONFIG["reg_activation"],
        "reg_entropy": INIT_CONFIG["reg_entropy"],
        "seed": INIT_CONFIG["seed"]
    })
    
    num_layers = gr.State(len(INIT_CONFIG["layers"]) + 1)
    layers = gr.State(INIT_CONFIG["layers"])
    layers_init = gr.State(INIT_CONFIG["layers"])

    num_layers.change(num_layers_change, [num_layers, layers], layers_init)
    layers.change(lambda x: x, layers, layers_init)

    with gr.Row(elem_id="interface-container"):
        with gr.Column(scale=0, min_width=1202):
            gr.Markdown("## KOLMOGOROV-ARNOLD NETWORK PLAYGROUND")
            with gr.Row(equal_height=True, elem_id="shadow_block"):
                run_btn = gr.Button("RUN", min_width=110, size="lg", scale=0, variant="primary")
                restart_btn = gr.Button("RESET", min_width=110, size="lg", scale=0)
                lr = gr.Dropdown(
                    [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                    value=INIT_CONFIG["lr"], allow_custom_value=True,
                    label="Learning rate"
                )
                optimizer = gr.Dropdown(
                    OPTIMIZERS, value=INIT_CONFIG["optimizer"], label="Optimizer", interactive=False)
                residual_fn = gr.Dropdown(
                    RESIDUAL_FN_LIST, value=INIT_CONFIG["residual_fn"], label="Residual function")
                reg_activation = gr.Dropdown(
                    [0, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                    value=INIT_CONFIG["reg_activation"], allow_custom_value=True,
                    label="Regularize activation"
                )
                reg_entropy = gr.Dropdown(
                    [0, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                    value=INIT_CONFIG["reg_entropy"], allow_custom_value=True,
                    label="Regularize entropy"
                )

            with gr.Row(equal_height=True):
                with gr.Column(scale=0, min_width=260):
                    with gr.Column(scale=1, elem_id="shadow_block"):
                        gr.Markdown("### DATA")
                        data_fn = gr.Dropdown(DATA_FN_LIST, label="Formula", container=False)
                        train_test_ratio = gr.Slider(
                            value=INIT_CONFIG["train_test_ratio"],
                            minimum=0.1, maximum=0.9, step=0.1,
                            label="Train-test ratio"
                        )
                        noise = gr.Slider(
                            value=INIT_CONFIG["noise"],
                            minimum=0, maximum=0.5, step=0.01,
                            label="Noise"
                        )
                        batch_size = gr.Slider(
                            value=INIT_CONFIG["batch_size"],
                            minimum=10, maximum=200, step=10,
                            label="Batch size"
                        )

                    with gr.Column(scale=1, elem_id="shadow_block"):
                        gr.Markdown("### BASIS FUNCTIONS")
                        gr.Dropdown(["B-spline", "else"], container=False)
                        spline_order = gr.Slider(
                            value=INIT_CONFIG["spline_order"],
                            minimum=0, maximum=8, step=1,
                            label="B-spline order"
                        )
                        grid_size = gr.Slider(
                            value=INIT_CONFIG["grid_size"],
                            minimum=1, maximum=20, step=1,
                            label="Grid size"
                        )

                with gr.Column(scale=0, elem_id="shadow_block", min_width=650):                  
                    @gr.render(inputs=layers_init)
                    def render_count(layers_init):
                        n = len(layers_init)
                        render_layers = []

                        with gr.Row(height=20.8, equal_height=True):
                            gr.Markdown("### KAN ")
                            gr.Markdown(" ")
                            add_layer_btn = gr.Button("+", min_width=35, size="md", scale=0, interactive=n < 4)
                            remove_layer_btn = gr.Button("-", min_width=35, size="md", scale=0, interactive=n > 1)
                            gr.Markdown(f"{n+1} LAYERS")
                            gr.Markdown(" ")
                            gr.Markdown(" ")
                            
                            add_layer_btn.click(lambda x: x + 1, num_layers, num_layers)
                            remove_layer_btn.click(lambda x: x - 1, num_layers, num_layers)
                        
                        with gr.Row(equal_height=True, height=38):
                            gr.Markdown(" ")
                            for i in layers_init:
                                num = gr.Number(
                                    value=i, minimum=1, maximum=4, step=1, precision=0,
                                    min_width=70, container=False
                                )
                                render_layers.append(num)
                            gr.Markdown("hidden <br /> nodes")

                        @gr.on(inputs=render_layers, outputs=layers)
                        def refresh_layers(*args):
                            return list(args)

                    kan_plot = gr.Plot(show_label=False)
                    num_params = gr.Markdown("Total number of parameters", height=31)

                with gr.Column(scale=0, min_width=260):
                    with gr.Column(scale=1, elem_id="shadow_block"):
                        gr.Markdown("### EPOCH")
                        epoch = gr.Markdown("0")

                    with gr.Column(scale=1, elem_id="shadow_block"):
                        gr.Markdown("### OUTPUT")
                        with gr.Row():
                            with gr.Column(min_width=150, scale=0):
                                gr.Markdown(f"Test RMSE loss <br />Train RMSE loss<br /> Regularization penalty")
                            losses = gr.Markdown("", rtl=True)
                        
                        loss_plot = gr.LinePlot(
                            x="x", y="y", color="data",
                            x_title="epoch", y_title="loss"
                        )

                    with gr.Column(scale=1, elem_id="shadow_block"):
                        gr.Markdown("### SEED")
                        seed = gr.Slider(
                            value=INIT_CONFIG["seed"],
                            minimum=0, maximum=10000, step=1,
                            label="Random seed"
                        )
                    
        with gr.Column(scale=0, min_width=805, elem_id="theory_container"):
            gr.Markdown(
                theory.text_1,
                sanitize_html=False,
                latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                  {"left": "$", "right": "$", "display": False}]
            )

            gr.Image(value="kan_plot.png", height=400, 
                     show_label=False, container=False, show_download_button=False)

            gr.Markdown(
                theory.text_2,
                sanitize_html=False,
                latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                  {"left": "$", "right": "$", "display": False}]
            )

            gr.Image(value="bsplines_plot.png", show_label=False, container=False,
                     show_download_button=False, interactive=False, show_fullscreen_button=False)

            gr.Markdown(
                theory.text_3,
                sanitize_html=False,
                latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                  {"left": "$", "right": "$", "display": False}]
            )

            representation = gr.Markdown("", label="Representation", visible=False)
            layers_init_str = gr.Text("", label="KAN", visible=False)

            def example_processing(data_fn, representation, layers_init_str, lr, reg_activation, reg_entropy):
                if representation == r"$$\frac{(x+y)^2}{2}-\frac{(x^2+y^2)}{2}$$":
                    batch_size = 70
                    seed = 2
                elif representation == r"$$\frac{(x+y)^2}{4}-\frac{(x-y)^2}{4}$$":
                    batch_size = 90
                    seed = 2
                else:
                    batch_size = 50
                    seed = 42
                return (list(map(int, layers_init_str[3:-3].split(","))),
                        "SiLU", 0.7, 0.03, batch_size, 3, 5, seed)


            examples = gr.Examples(
                examples=[
                    [DATA_FN_LIST[0], r"$$e^{\large\frac{\sin(\pi x) + y^2}{2}} - 1$$", "[2,1,1]", 0.01, 0, 0],
                    [DATA_FN_LIST[0], r"$$e^{\large\frac{\sin(\pi x) + y^2}{2}} - 1$$", "[2,3,1,1]", 0.03, 0.01, 0.01],
                    [DATA_FN_LIST[1], r"$$e^{\ln(2x+3)-\ln(2y+3)}-1$$", "[2,1,1]", 0.01, 0, 0],
                    [DATA_FN_LIST[1], r"$$e^{\ln(2x+3)-\ln(2y+3)}-1$$", "[2,3,2,1]", 0.03, 0.01, 0.01],
                    [DATA_FN_LIST[4], r"$$\frac{(x+y)^2}{2}-\frac{(x^2+y^2)}{2}$$", "[2,2,1]", 0.1, 0, 0],
                    [DATA_FN_LIST[4], r"$$\frac{(x+y)^2}{4}-\frac{(x-y)^2}{4}$$", "[2,2,1]", 0.1, 0, 0],
                ],
                inputs=[data_fn, representation, layers_init_str, lr, reg_activation, reg_entropy],
                fn=example_processing,
                outputs=[layers_init, residual_fn, train_test_ratio, noise, batch_size, spline_order, grid_size, seed],
                run_on_click=True, cache_examples=False,
            )

            gr.Markdown(theory.references, sanitize_html=False)


    gr.on(
        triggers=[optimizer.change, lr.change, reg_activation.change, reg_entropy.change],
        fn=update_learning_parameters,
        inputs=[optimizer, lr, reg_activation, reg_entropy, state], outputs=state,
    )

    gr.on(
        triggers=[layers.change, grid_size.change, spline_order.change, residual_fn.change, restart_btn.click],
        fn=stop_train, inputs=[state], outputs=[run_btn, epoch]
    ).success(
        update_model_parameters, [layers, grid_size,spline_order, residual_fn, state], state
    ).success(update_model, state, [kan_plot, epoch, losses, loss_plot, num_params])

    gr.on(
        triggers=[data_fn.change, train_test_ratio.change, noise.change, batch_size.change, seed.change, demo.load],
        fn=stop_train, inputs=state, outputs=[run_btn, epoch]
    ).success(
        update_data, [data_fn, train_test_ratio, noise, batch_size, state, seed], state,
    ).success(update_model, state, [kan_plot, epoch, losses, loss_plot, num_params])

    run_btn.click(run_pause_train, state, run_btn)
    run_btn.click(train, state, [kan_plot, epoch, losses, loss_plot])


demo.launch()
