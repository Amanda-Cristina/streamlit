from dataclasses import dataclass
import yaml

import pandas as pd


@dataclass
class APPCONFIGS:
    nome: str
    versao: float
    descricao: str
    links_referencias: list
    parametros: dict
    tests: dict

    def get_tests(self):
        """
        Retorna atributo tests tratado convertento dados
        no campo dataframe para Pandas

        Returns:
        ----------
        res:dict
            Dicionário com dados para serem inputados na
            função main
        """
        res = {}
        for nome, valor in self.tests["dataframe"].items():
            res[nome] = pd.read_csv(valor)
        for nome, valor in self.tests["inputs"].items():
            res[nome] = valor
        return res


def load_configs(path: str):
    """
    Carrega dados do YAML e os converte em uma
    dataclasse python
    """
    with open(path, "r") as stream:
        parsed_yaml = yaml.safe_load(stream)
        return APPCONFIGS(**parsed_yaml)


if __name__ == "__main__":
    cfg = load_configs(path="rfv/configs.yaml")
    print(cfg.__dict__)
