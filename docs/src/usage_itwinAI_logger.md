
# Usage with ItwinAI Logger

<div style="display: flex; align-items: center; background-color: #cc3300; color: #333; border: 5px solid #cc3300; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">‼</span>
    <span style="margin-left: 35px; padding: 5px; background-color: white; border-radius: 5px; width: 100%">The ItwinAI logger is supported only by yProv4ML version 1.0, check the installation process. </span>
</div>

This section provides an example of how to use Prov4ML with PyTorch Lightning.

In any lightning module the calls to `train_step`, `validation_step`, and `test_step` can be overridden to log the necessary information.

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>

```python
trainer = L.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=EPOCHS,
    enable_checkpointing=False, 
    log_every_n_steps=1, 
    logger=[prov4ml.ProvMLItwinAILogger()],
)
```

<hr style="border: 2px solid #009B77; margin: 20px 0;">

<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="usage_lightning.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="reproducible_example.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>