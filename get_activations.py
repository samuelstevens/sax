"""
Gets activations for a ViT's image representation and stores them as floating-point arrays on disk so they can be used as training data for SAEs.
"""
import tyro
import dataclasses

@dataclasses.dataclass(frozen=True)
class Args:
    pass

def main(args: Args):
    pass

if __name__ == '__main__':
    main(tyro.cli(Args))
