from entity_gym.runner import CliRunner

from enn_zoo.microrts import GymMicrorts

if __name__ == "__main__":
    CliRunner(GymMicrorts()).run()
