import click


@click.command()
@click.option("--device", type=str, default="cpu")
def main(device: str):
    pass  # TODO


main()
