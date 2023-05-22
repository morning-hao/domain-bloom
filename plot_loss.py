import json
import argparse
import matplotlib.pyplot as plt


def plot_log_history(json_file, save_path):
    # 读取JSON文件
    with open(json_file) as f:
        data = json.load(f)

    # 提取"log_history"中的数据
    log_history = data["log_history"]

    # 提取所需的信息
    epochs = [log['epoch'] for log in log_history if 'loss' in log]
    loss_steps = [log['step'] for log in log_history if 'loss' in log]
    losses = [log['loss'] for log in log_history if 'loss' in log]
    learning_rate_steps = [log['step'] for log in log_history if 'learning_rate' in log]
    learning_rates = [log['learning_rate'] for log in log_history if 'learning_rate' in log]
    eval_steps = [log['step'] for log in log_history if 'eval_loss' in log]
    eval_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]

    # 创建一个新的图像
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # 绘制Loss
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss', color='#1f77b4')
    ax1.plot(loss_steps, losses, label='Training loss', color='#1f77b4')
    ax1.plot(eval_steps, eval_losses, label='Evaluation loss', color='#ff7f0e')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.legend(loc='upper left')

    # 添加第二个y轴，用于绘制Learning rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='#2ca02c')
    ax2.plot(learning_rate_steps, learning_rates, label='Learning rate', color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')

    fig.tight_layout()  # 避免标签重叠
    plt.title('Loss and Learning Rate over Steps')

    # 保存图像
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, default="", required=True, help="The path of the JSON file to be processed.")
    parser.add_argument("--save_path", type=str,  default="output_dir/plot_loss.png", help="The path where the output image will be saved.")
    args = parser.parse_args()
    plot_log_history(args.json_file, args.save_path)
