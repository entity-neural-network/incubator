import functools

import numpy as np
import opensimplex

from enn_zoo.griddly_env.level_generators.level_generator import LevelGenerator


class CrafterLevelGenerator(LevelGenerator):
    """
    Pretty much this: https://github.com/danijar/crafter/blob/main/crafter/worldgen.py
    """

    def __init__(self, seed, width, height, num_players):
        self._width = width
        self._height = height
        self._num_players = num_players
        self._walkable = {"G", "P", "S"}
        self._random = np.random.RandomState(seed)

    def _get_level_string(self, world):
        level_string = ""
        for h in range(0, self._height):
            for w in range(0, self._width):
                level_string += world[w, h].ljust(4)
            level_string += "\n"

        return level_string

    def generate(self) -> str:
        """
        Generate a crafter-style level and return the griddly level string for it
        :return:
        """
        world = np.chararray(
            shape=(self._width, self._height), unicode=True, itemsize=4
        )
        tunnels = np.zeros((self._width, self._height), np.bool)

        simplex = opensimplex.OpenSimplex(seed=self._random.randint(0, 2**31 - 1))
        for x in range(self._width):
            for y in range(self._height):
                self._set_material(world, (x, y), simplex, tunnels)

        # Add the players to random grass locations:
        players = []
        not_tunnels = set(map(tuple, np.array(np.where(tunnels == False)).T))
        grass = set(map(tuple, np.array(np.where(world == "G")).T))

        possible_player_locations = list(not_tunnels & grass)

        for p in range(self._num_players):
            players.append(
                self._place_player(world, possible_player_locations, players)
            )

        for x in range(self._width):
            for y in range(self._height):
                self._set_object(world, (x, y), players, tunnels)

        return self._get_level_string(world)

    def _place_player(self, world, possible_locations, players):

        player_location_valid = False

        while not player_location_valid:
            player_pos = possible_locations[
                self._random.choice(len(possible_locations))
            ]
            dist = self._get_min_player_distance(player_pos, players)
            possible_locations.remove(player_pos)

            if dist > 10:
                world[player_pos[0], player_pos[1]] = f"p{len(players)+1}/G"
                return player_pos

    def _set_material(self, world, pos, simplex, tunnels):
        x, y = pos
        start_x, start_y = self._random.randint((0, 0), (self._width, self._height))
        simplex = functools.partial(self._simplex, simplex)
        uniform = self._random.uniform
        start = 4 - np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
        start += 2 * simplex(x, y, 8, 3)
        start = 1 / (1 + np.exp(-start))
        water = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
        water -= 2 * start
        mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
        mountain -= 4 * start + 0.3 * water
        if start > 0.5:
            world[x, y] = "G"
        elif mountain > 0.15:
            if simplex(x, y, 6, 7) > 0.15 and mountain > 0.3:  # cave
                world[x, y] = "P"
            elif simplex(2 * x, y / 5, 7, 3) > 0.4:
                world[x, y] = "P"
                tunnels[x, y] = True
            elif simplex(x / 5, 2 * y, 7, 3) > 0.4:
                world[x, y] = "P"
                tunnels[x, y] = True
            elif simplex(x, y, 1, 8) > 0 and uniform() > 0.85:
                world[x, y] = "c"
            elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.75:
                world[x, y] = "i"
            elif mountain > 0.18 and uniform() > 0.994:
                world[x, y] = "d"
            elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.35:
                world[x, y] = "L"
            else:
                world[x, y] = "s"
        elif 0.25 < water <= 0.35 and simplex(x, y, 4, 9) > -0.2:
            world[x, y] = "S"
        elif 0.3 < water:
            world[x, y] = "W"
        else:  # grassland
            if simplex(x, y, 5, 7) > 0 and uniform() > 0.8:
                world[x, y] = "T"
            else:
                world[x, y] = "G"

    def _get_min_player_distance(self, pos, players):
        dist = np.inf
        for player_pos in players:
            cur_dist = np.linalg.norm(np.array(player_pos) - np.array(pos))
            if cur_dist < dist:
                dist = dist
        return dist

    def _set_object(self, world, pos, players, tunnels):
        x, y = pos
        uniform = self._random.uniform

        dist = self._get_min_player_distance(pos, players)

        material = world[x, y]
        if material not in self._walkable:
            pass
        elif dist > 3 and material == "G" and uniform() > 0.985:  # cow
            world[x, y] = f"#/{material}"
        elif dist > 10 and uniform() > 0.993:  # zombie
            world[x, y] = f"!/{material}"
        elif material == "P" and tunnels[x, y] and uniform() > 0.95:  # skeleton
            world[x, y] = f"@/{material}"

    def _simplex(self, simplex, x, y, z, sizes, normalize=True):
        if not isinstance(sizes, dict):
            sizes = {sizes: 1}
        value = 0
        for size, weight in sizes.items():
            value += weight * simplex.noise3(x / size, y / size, z)
        if normalize:
            value /= sum(sizes.values())
        return value
