import os

from torch import poisson
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import numpy as np
import scipy.spatial.distance as ssd
from gym import spaces
from gym.utils import seeding
from .._utils import Agent
import pygame
import math


class Archea(Agent):

    def __init__(self, idx, radius, n_sensors, sensor_range, max_accel, dist_action=False,
                 group=None, speed_features=True, indicator_dim=3, group_num=1):
        self._idx = idx
        self._radius = radius
        self._n_sensors = n_sensors
        self._sensor_range = sensor_range
        self._max_accel = max_accel
        self.dist_action = dist_action
        # Number of observation coordinates from each sensor
        self._sensor_obscoord = 5 + group_num - 1
        if speed_features:
            self._sensor_obscoord += 3 + group_num - 1
        self._sensor_obs_coord = self._n_sensors * self._sensor_obscoord
        self._obs_dim = self._sensor_obs_coord + 2 + indicator_dim + 4 
        # +1 for is_colliding_evader, +1 for is_colliding_poison
        # +4 for self position and self velocity

        self._position = None
        self._velocity = None

        # Generate self._n_sensors angles, evenly spaced from 0 to 2pi
        # We generate 1 extra angle and remove it because linspace[0] = 0 = 2pi = linspace[-1]
        angles = np.linspace(0., 2. * np.pi, self._n_sensors + 1)[:-1]
        # Convert angles to x-y coordinates
        sensor_vectors = np.c_[np.cos(angles), np.sin(angles)]
        self._sensors = sensor_vectors

        self._group = group
        self._visible = True

    @property
    def observation_space(self):
        return spaces.Box(low=np.float32(-np.sqrt(2)), high=np.float32(2 * np.sqrt(2)), shape=(self._obs_dim,), dtype=np.float32)

    @property
    def action_space(self):
        if self.dist_action:
            return spaces.Discrete(9)
        else:
            return spaces.Box(low=np.float32(-self._max_accel), high=np.float32(self._max_accel), shape=(2,), dtype=np.float32)

    @property
    def position(self):
        assert self._position is not None
        return self._position

    @property
    def velocity(self):
        assert self._velocity is not None
        return self._velocity
    
    @property
    def visible(self):
        return self._visible
    
    def hide(self):
        self._visible = False
    
    def reveal(self):
        self._visible = True

    def set_position(self, pos):
        assert pos.shape == (2,)
        self._position = pos

    def set_velocity(self, velocity):
        assert velocity.shape == (2,)
        self._velocity = velocity

    @property
    def sensors(self):
        assert self._sensors is not None
        return self._sensors

    def sensed(self, object_coord, object_radius, same=False, visible_list=None):
        """Whether object would be sensed by the pursuers"""
        relative_coord = object_coord - np.expand_dims(self.position, 0)
        # Projection of object coordinate in direction of sensor
        sensorvals = self.sensors.dot(relative_coord.T)
        # Set sensorvals to np.inf when object should not be seen by sensor
        distance_squared = (relative_coord**2).sum(axis=1)[None, :]
        sensorvals[
            (sensorvals < 0)    # Wrong direction (by more than 90 degrees in both directions)
            | (sensorvals - object_radius > self._sensor_range)         # Outside sensor range
            | (distance_squared - sensorvals**2 > object_radius**2)     # Sensor does not intersect object
        ] = np.inf
        if same:
            # Set sensors values for sensing the current object to np.inf
            sensorvals[:, self._idx - 1] = np.inf
        if visible_list is not None:
            for i in range(len(visible_list)):
                if not visible_list[i]:
                    sensorvals[:, i] = np.inf
        return sensorvals

    def sense_barriers(self, min_pos=0, max_pos=1):
        sensor_vectors = self.sensors * self._sensor_range
        sensor_endpoints = sensor_vectors + self.position

        # Clip sensor lines on the environment's barriers.
        # Note that any clipped vectors may not be at the same angle as the original sensors
        clipped_endpoints = np.clip(sensor_endpoints, min_pos, max_pos)

        # Extract just the sensor vectors after clipping
        clipped_vectors = clipped_endpoints - self.position

        # Find the ratio of the clipped sensor vector to the original sensor vector
        # Scaling the vector by this ratio will limit the end of the vector to the barriers
        ratios = np.divide(clipped_vectors, sensor_vectors, out=np.ones_like(clipped_vectors),
                           where=np.abs(sensor_vectors) > 0.00000001)

        # Find the minimum ratio (x or y) of clipped endpoints to original endpoints
        minimum_ratios = np.amin(ratios, axis=1)

        # Convert to 2d array of size (n_sensors, 1)
        sensor_values = np.expand_dims(minimum_ratios, 0)

        # Set values beyond sensor range to infinity
        does_sense = minimum_ratios < (1.0 - 1e-4)
        does_sense = np.expand_dims(does_sense, 0)
        sensor_values[np.logical_not(does_sense)] = np.inf

        # Convert -0 to 0
        sensor_values[sensor_values == -0] = 0

        return sensor_values.T


class FoodCollector():

    def __init__(self, n_pursuers=3, n_evaders=3, n_poison=10, n_coop=1, n_sensors=30, sensor_range=0.2,
                 radius=0.015, poison_scale=0.75, obstacle_radius=0.2, obstacle_coord=np.array([0.5, 0.5]),
                 pursuer_max_accel=0.01, evader_speed=0.005, poison_speed=0.005, poison_reward=-1.0,
                 food_reward=10.0, food_revive=False, encounter_reward=0.01, thrust_penalty=-0.5, local_ratio=1.0,
                 speed_features=False, max_cycles=500, comm=False, dist_action=False, soft_reward=False,
                 comm_freq=1, use_groudtruth=False, smart_comm=False, food_alive_penalty=0.5, 
                 window_size=2, **kwargs):
        """
        n_pursuers: number of pursuing archea (agents), the same as number of groups
        n_evaders: number of evader archea per group
        n_poison: number of poison archea
        n_coop: number of pursuing archea (agents) that must be touching food at the same time to consume it
        n_sensors: number of sensors on all pursuing archea (agents)
        sensor_range: length of sensor dendrite on all pursuing archea (agents)
        radius: archea base radius. Pursuer: radius, evader: 2 x radius
        poison_scale: scale of poison, radius of poison = poison_scale x radius
        obstacle_radius: radius of obstacle object
        obstacle_coord: coordinate of obstacle object. Can be set to `None` to use a random location
        pursuer_max_accel: pursuer archea maximum acceleration (maximum action size)
        evader_speed: evading archea speed
        poison_speed: poison archea speed
        poison_reward: reward for pursuer consuming a poison object (typically negative)
        food_reward:reward for pursuers consuming an evading archea
        encounter_reward: reward for a pursuer colliding with an evading archea
        thrust_penalty: scaling factor for the negative reward used to penalize large actions
        local_ratio: Proportion of reward allocated locally vs distributed globally among all agents
        speed_features: toggles whether pursuing archea (agent) sensors detect speed of other archea
        max_cycles: After max_cycles steps all agents will return done
        comm: whether to use communication
        dist_action: whether to use discrete actions
        soft_reward: whether to use a softer reward function (not implemented yet)
        """
        self.n_groups = n_pursuers
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders * self.n_groups
        self.n_evaders_pergroup = n_evaders
        self.n_coop = n_coop
        self.n_poison = n_poison
        self.dist_action = dist_action
        self.soft_reward = soft_reward
        self.use_groudtruth = use_groudtruth
        
        self.obstacle_radius = obstacle_radius
        obstacle_coord = np.array(obstacle_coord)
        self.initial_obstacle_coord = np.random.uniform(0, 1, 2) if obstacle_coord is None else obstacle_coord
        self.pursuer_max_accel = pursuer_max_accel
        self.evader_speed = evader_speed
        self.poison_speed = poison_speed
        self.radius = radius
        self.n_sensors = n_sensors
        self.sensor_range = np.ones(self.n_pursuers) * min(sensor_range, (math.ceil(math.sqrt(2) * 100) / 100.0))
        self.poison_reward = poison_reward
        self.food_reward = food_reward
        self.thrust_penalty = thrust_penalty
        self.encounter_reward = encounter_reward
        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = [None for _ in range(self.n_pursuers)]
        self.comm = comm
        print("communicating", self.comm)
        self.last_comm = [None for _ in range(self.n_pursuers)]
        self.comm_freq = comm_freq
        self.smart_comm = smart_comm
        self.food_revive = food_revive
        if self.food_revive:
            food_alive_penalty = 0

        self.n_obstacles = 1
        self.local_ratio = local_ratio
        self._speed_features = speed_features
        self.poison_scale = poison_scale
        self.max_cycles = max_cycles
        self.food_alive_penalty = food_alive_penalty
        self.seed()
        # TODO: Look into changing hardcoded radius ratios
        self._pursuers = [
            Archea(pursuer_idx + 1, self.radius, self.n_sensors, sensor_range, self.pursuer_max_accel,
                   speed_features=self._speed_features, group=pursuer_idx, indicator_dim=self.n_pursuers,
                   group_num=self.n_pursuers, dist_action=self.dist_action)
            for pursuer_idx in range(self.n_pursuers)
        ]
        self._evaders = []
        for g in range(self.n_groups):
            self._evaders += [
                Archea(evader_idx + 1, self.radius * 2, self.n_pursuers, 0, self.evader_speed, group=g)
                for evader_idx in range(self.n_evaders_pergroup)
            ]
        self._poisons = [
            Archea(poison_idx + 1, self.radius * self.poison_scale, self.n_poison, 0, self.poison_speed)
            for poison_idx in range(self.n_poison)
        ]

        self.colors = [
            (192, 64, 64),
            (64, 192, 64),
            (64, 64, 192),
            (192, 192, 64),
            (192, 64, 192),
            (64, 192, 192),
        ]

        self.num_agents = self.n_pursuers
        self.action_space = [agent.action_space for agent in self._pursuers]
        self.observation_space = [
            agent.observation_space for agent in self._pursuers]
        if self.use_groudtruth:
            comm_dim = n_evaders * 2 
        else:
            comm_dim = (self.n_sensors + 2) * (self.n_groups - 1)
        self.communication_space = [
            spaces.Box(low=-1, high=1, shape=(comm_dim,), dtype=np.float32)
            for _ in self._pursuers
        ]
        print("observation space", self.observation_space)
        print("communication space", self.communication_space)

        self.renderOn = False
        self.pixel_scale = 30 * 25
        self.window_size = window_size

        self.cycle_time = 1.0
        self.frames = 0
        self.reset()

    def close(self):
        if self.renderOn:
            # pygame.event.pump()
            pygame.display.quit()
            pygame.quit()

    @property
    def agents(self):
        return self._pursuers

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _generate_coord(self, radius):
        coord = self.np_random.rand(2)
        # Create random coordinate that avoids obstacles
        while ssd.cdist(coord[None, :], self.obstacle_coords) <= radius * 2 + self.obstacle_radius:
            coord = self.np_random.rand(2)
        return coord

    def reset(self):
        self.frames = 0
        # Initialize obstacles
        if self.initial_obstacle_coord is None:
            # Generate obstacle positions in range [0, 1)
            self.obstacle_coords = self.np_random.rand(self.n_obstacles, 2)
        else:
            self.obstacle_coords = self.initial_obstacle_coord[None, :]
        # Set each obstacle's velocity to 0
        # TODO: remove if obstacles should never move
        self.obstacle_speeds = np.zeros((self.n_obstacles, 2))

        # Initialize pursuers
        for pursuer in self._pursuers:
            pursuer.set_position(self._generate_coord(pursuer._radius))
            pursuer.set_velocity(np.zeros(2))

        # Initialize evaders
        for evader in self._evaders:
            evader.set_position(self._generate_coord(evader._radius))
            # Generate velocity such that speed <= self.evader_speed
            velocity = self.np_random.rand(2) - 0.5
            speed = np.linalg.norm(velocity)
            if speed > self.evader_speed:
                # Limit speed to self.evader_speed
                velocity = velocity / speed * self.evader_speed
            evader.set_velocity(velocity)
            evader.reveal()

        # Initialize poisons
        for poison in self._poisons:
            poison.set_position(self._generate_coord(poison._radius))
            # Generate both velocity components from range [-self.poison_speed, self.poison_speed)
            # Generate velocity such that speed <= self.poison_speed
            velocity = self.np_random.rand(2) - 0.5
            speed = np.linalg.norm(velocity)
            if speed > self.poison_speed:
                # Limit speed to self.poison_speed
                velocity = velocity / speed * self.poison_speed
            poison.set_velocity(velocity)

        self.last_comm = [np.zeros(self.communication_space[0].shape[0]) for _ in self._pursuers]

        rewards = np.zeros(self.n_pursuers)
        sensor_features, collided_pursuer_evader, collided_pursuer_poison, rewards \
            = self.collision_handling_subroutine(rewards, True)
        
        obs_list = self.observe_list(
            sensor_features, collided_pursuer_evader, collided_pursuer_poison)
        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = obs_list
        return obs_list[0]

    def _caught(self, is_colliding_x_y):
        """ Check whether collision results in catching the object
        """
        # Number of collisions for each y
        n_collisions = is_colliding_x_y.sum(axis=0)
        # List of y that have been caught
        caught_y = np.where(n_collisions >= 1)[0]

        # Boolean array indicating which x caught any y in caught_y
        did_x_catch_y = is_colliding_x_y[:, caught_y]
        # List of x that caught corresponding y in caught_y
        x_caught_y = np.where(did_x_catch_y >= 1)[0]

        return caught_y, x_caught_y
    
    def _caught_bygroup(self, is_colliding_x_y):
        """ Check whether collision results in catching the object
        by_group: whether to check in-group collision only
        """
        m, _ = is_colliding_x_y.shape
        caught_y, x_caught_y = np.array([], dtype=int), np.array([], dtype=int)
        for i in range(m):
            collide_all_group = np.where(is_colliding_x_y[i,:] == True)[0]
            collide_in_group_idx = np.where(collide_all_group//self.n_evaders_pergroup == i)[0]
            if len(collide_in_group_idx)>0:
                collide_in_group_y = collide_all_group[collide_in_group_idx]
                caught_y = np.concatenate((caught_y,collide_in_group_y), axis=0)
                x_caught_y = np.concatenate((x_caught_y, np.array([i])), axis=0)

        return caught_y, x_caught_y

    def _closest_dist(self, closest_object_idx, input_sensorvals):
        """Closest distances according to `idx`"""
        sensorvals = []

        for pursuer_idx in range(self.n_pursuers):
            sensors = np.arange(self.n_sensors)         # sensor indices
            objects = closest_object_idx[pursuer_idx, ...]  # object indices
            sensorvals.append(input_sensorvals[pursuer_idx, ..., sensors, objects])

        return np.c_[sensorvals]

    def _extract_speed_features(self, object_velocities, object_sensorvals, sensed_mask):
        # sensed_mask is a boolean mask of which sensor values detected an object
        sensorvals = []
        for pursuer in self._pursuers:
            relative_speed = object_velocities - np.expand_dims(pursuer.velocity, 0)
            sensorvals.append(pursuer.sensors.dot(relative_speed.T))
        sensed_speed = np.c_[sensorvals]    # Speeds in direction of each sensor

        speed_features = np.zeros((self.n_pursuers, self.n_sensors))

        sensorvals = []
        for pursuer_idx in range(self.n_pursuers):
            sensorvals.append(
                sensed_speed[pursuer_idx, :, :][np.arange(self.n_sensors), object_sensorvals[pursuer_idx, :]]
            )
        # Set sensed values, all others remain 0
        speed_features[sensed_mask] = np.c_[sensorvals][sensed_mask]

        return speed_features

    def collision_handling_subroutine(self, rewards, is_last):
        # Stop pursuers upon hitting a wall
        for pursuer in self._pursuers:
            clipped_coord = np.clip(pursuer.position, 0, 1)
            clipped_velocity = pursuer.velocity
            # If x or y position gets clipped, set x or y velocity to 0 respectively
            clipped_velocity[pursuer.position != clipped_coord] = 0
            # Save clipped velocity and position
            pursuer.set_velocity(clipped_velocity)
            pursuer.set_position(clipped_coord)

        def rebound_particles(particles, n):
            collisions_particle_obstacle = np.zeros(n)
            # Particles rebound on hitting an obstacle
            for idx, particle in enumerate(particles):
                obstacle_distance = ssd.cdist(np.expand_dims(
                    particle.position, 0), self.obstacle_coords)
                is_colliding = obstacle_distance <= particle._radius + self.obstacle_radius
                collisions_particle_obstacle[idx] = is_colliding.sum()
                if collisions_particle_obstacle[idx] > 0:
                    # Rebound the particle that collided with an obstacle
                    velocity_scale = particle._radius + self.obstacle_radius - \
                        ssd.euclidean(particle.position, self.obstacle_coords)
                    pos_diff = particle.position - self.obstacle_coords[0]
                    new_pos = particle.position + velocity_scale * pos_diff
                    particle.set_position(new_pos)

                    collision_normal = particle.position - self.obstacle_coords[0]
                    # project current velocity onto collision normal
                    current_vel = particle.velocity
                    proj_numer = np.dot(current_vel, collision_normal)
                    cllsn_mag = np.dot(collision_normal, collision_normal)
                    proj_vel = (proj_numer / cllsn_mag) * collision_normal
                    perp_vel = current_vel - proj_vel
                    total_vel = perp_vel - proj_vel
                    particle.set_velocity(total_vel)

        rebound_particles(self._pursuers, self.n_pursuers)

        if is_last:
            rebound_particles(self._evaders, self.n_evaders)
            rebound_particles(self._poisons, self.n_poison)

        positions_pursuer = np.array([pursuer.position for pursuer in self._pursuers])
        positions_evader = np.array([evader.position for evader in self._evaders])
        positions_poison = np.array([poison.position for poison in self._poisons])

        # Find evader collisions
        distances_pursuer_evader = ssd.cdist(positions_pursuer, positions_evader)
        # Generate n_evaders x n_pursuers matrix of boolean values for collisions
        collisions_pursuer_evader_raw = distances_pursuer_evader <= np.asarray([
            pursuer._radius + evader._radius for pursuer in self._pursuers
            for evader in self._evaders
        ]).reshape(self.n_pursuers, self.n_evaders)

        visible_evader = np.tile(np.array([evader.visible for evader in self._evaders]), (self.n_groups,1))
        collisions_pursuer_evader = np.logical_and(collisions_pursuer_evader_raw, visible_evader)
        # Number of collisions depends on n_coop, how many are needed to catch an evader
        caught_evaders, pursuer_evader_catches = self._caught_bygroup(
            collisions_pursuer_evader)
        if not self.food_revive:
            for ce in caught_evaders:
                self._evaders[ce].hide()
        for g in pursuer_evader_catches:
            self.last_comm[g] = np.zeros(self.communication_space[0].shape[0])
        # if self.soft_reward:
        #     print("dist", distances_pursuer_evader)
        #     dist_caught = distances_pursuer_evader
        #     rewards[pursuer_evader_catches] -= self.food_reward

        # if np.any(collisions_pursuer_evader):
        #     print("visiable", visible_evader)
        #     print("collision matrix", collisions_pursuer_evader_raw)
        #     print("visible collision", collisions_pursuer_evader)
        #     print("caught evaders", caught_evaders)
        #     print("pursuer_evader_catches",pursuer_evader_catches)
        # if np.any(caught_evaders):
        #     print("caught evaders", caught_evaders)

        # Find poison collisions
        distances_pursuer_poison = ssd.cdist(positions_pursuer, positions_poison)
        collisions_pursuer_poison = distances_pursuer_poison <= np.asarray([
            pursuer._radius + poison._radius for pursuer in self._pursuers
            for poison in self._poisons
        ]).reshape(self.n_pursuers, self.n_poison)

        caught_poisons, pursuer_poison_collisions = self._caught(
            collisions_pursuer_poison)

        # Find sensed obstacles
        sensorvals_pursuer_obstacle = np.array(
            [pursuer.sensed(self.obstacle_coords, self.obstacle_radius) for pursuer in self._pursuers])

        # Find sensed barriers
        sensorvals_pursuer_barrier = np.array(
            [pursuer.sense_barriers(max_pos=self.window_size) for pursuer in self._pursuers])

        # Find sensed evaders, separately for different groups
        sensorvals_pursuer_evader_group = []
        visible_evader_arr = np.array([evader.visible for evader in self._evaders])
        for g in range(self.n_groups):
            visible_list = visible_evader_arr[g*self.n_evaders_pergroup:(g+1)*self.n_evaders_pergroup]
            sensorvals_pursuer_evader = np.array(
                [pursuer.sensed(positions_evader[g*self.n_evaders_pergroup:(g+1)*self.n_evaders_pergroup], 
                self.radius * 2, visible_list=visible_list) for pursuer in self._pursuers])
            sensorvals_pursuer_evader_group.append(sensorvals_pursuer_evader)

        # Find sensed poisons
        sensorvals_pursuer_poison = np.array(
            [pursuer.sensed(positions_poison, self.radius * self.poison_scale) for pursuer in self._pursuers])

        # Find sensed pursuers
        sensorvals_pursuer_pursuer = np.array(
            [pursuer.sensed(positions_pursuer, self.radius, same=True) for pursuer in self._pursuers])

        # Collect distance features
        def sensor_features(sensorvals):
            closest_idx_array = np.argmin(sensorvals, axis=2)
            closest_distances = self._closest_dist(closest_idx_array, sensorvals)
            finite_mask = np.isfinite(closest_distances)
            sensed_distances = np.ones((self.n_pursuers, self.n_sensors))
            sensed_distances[finite_mask] = closest_distances[finite_mask]
            return sensed_distances, closest_idx_array, finite_mask

        obstacle_distance_features, _, _ = sensor_features(sensorvals_pursuer_obstacle)
        barrier_distance_features, _, _ = sensor_features(sensorvals_pursuer_barrier)
        evader_distance_features, closest_evader_idx, evader_mask = [], [], []
        for g in range(self.n_groups):
            edf, ec, em = sensor_features(sensorvals_pursuer_evader_group[g])
            evader_distance_features.append(edf)
            closest_evader_idx.append(ec)
            evader_mask.append(em)
        poison_distance_features, closest_poison_idx, poison_mask = sensor_features(sensorvals_pursuer_poison)
        pursuer_distance_features, closest_pursuer_idx, pursuer_mask = sensor_features(sensorvals_pursuer_pursuer)

        # Collect speed features
        if self._speed_features:
            pursuers_speed = np.array([pursuer.velocity for pursuer in self._pursuers])
            evaders_speed = np.array([evader.velocity for evader in self._evaders])
            poisons_speed = np.array([poison.velocity for poison in self._poisons])

            evader_speed_features = []
            for g in range(self.n_groups):
                start = g*self.n_evaders_pergroup
                end = (g+1)*self.n_evaders_pergroup
                esf = self._extract_speed_features(evaders_speed[start:end], closest_evader_idx[g], evader_mask[g])
                evader_speed_features.append(esf)

            poison_speed_features = self._extract_speed_features(poisons_speed, closest_poison_idx, poison_mask)
            pursuer_speed_features = self._extract_speed_features(pursuers_speed, closest_pursuer_idx, pursuer_mask)

        # Process collisions
        # If object collided with required number of players, reset its position and velocity
        # Effectively the same as removing it and adding it back
        def reset_caught_objects(caught_objects, objects, speed):
            if caught_objects.size:
                for object_idx in caught_objects:
                    objects[object_idx].set_position(
                        self._generate_coord(objects[object_idx]._radius))
                    # Generate both velocity components from range [-self.evader_speed, self.evader_speed)
                    objects[object_idx].set_velocity(
                        (self.np_random.rand(2,) - 0.5) * 2 * speed)

        # remove food if it is caught
        # but if food can revive, reset them to other places
        if self.food_revive:
            reset_caught_objects(caught_evaders, self._evaders, self.evader_speed)
        reset_caught_objects(caught_poisons, self._poisons, self.poison_speed)
        
        pursuer_evader_encounters, pursuer_evader_encounter_matrix = self._caught_bygroup(
            collisions_pursuer_evader)

        # Update reward based on these collisions
        rewards[pursuer_evader_catches] += self.food_reward
        rewards[pursuer_poison_collisions] += self.poison_reward
        rewards[pursuer_evader_encounter_matrix] += self.encounter_reward

        if self.frames % self.comm_freq == 0:
            if self.use_groudtruth:
                self.send_groudtruth()
            else:
                self.send_message(evader_distance_features)

        # Add features together
        if self._speed_features:
            evader_distance_features = np.concatenate(evader_distance_features, axis=1)
            evader_speed_features = np.concatenate(evader_speed_features, axis=1)
            sensorfeatures = np.c_[
                obstacle_distance_features, barrier_distance_features,
                evader_distance_features, evader_speed_features,
                poison_distance_features, poison_speed_features,
                pursuer_distance_features, pursuer_speed_features
            ]
        else:
            evader_distance_features = np.concatenate(evader_distance_features, axis=1)
            sensorfeatures = np.c_[
                obstacle_distance_features,
                barrier_distance_features,
                evader_distance_features,
                poison_distance_features,
                pursuer_distance_features
            ]

        return sensorfeatures, collisions_pursuer_evader, collisions_pursuer_poison, rewards

    def observe_list(self, sensor_feature, is_colliding_evader, is_colliding_poison):
        obslist = []
        for pursuer_idx in range(self.n_pursuers):
            one_hot = np.zeros(self.n_pursuers)
            one_hot[pursuer_idx] = 1.0
            position = self._pursuers[pursuer_idx].position
            velocity = self._pursuers[pursuer_idx].velocity
            obslist.append(
                np.concatenate([
                    sensor_feature[pursuer_idx, ...].ravel(), [
                        float((is_colliding_evader[pursuer_idx, :]).sum() > 0), float((
                            is_colliding_poison[pursuer_idx, :]).sum() > 0)
                    ], one_hot, position, velocity
                ]))
        return obslist
    
    def send_message(self, evader_distance_features):
        positions = np.zeros((self.n_pursuers, 2))
        for p in range(self.n_pursuers):
            positions[p,:] = self._pursuers[p].position
        l = int(self.communication_space[0].shape[0] / (self.n_pursuers - 1))

        for g in range(self.n_groups):
            row_mask = np.ones(self.n_groups, bool)
            row_mask[g] = False
            message = evader_distance_features[g][row_mask,:]
            senders = positions[row_mask,:]
            
            if self.smart_comm:
                # send only when valid
                for m in range(len(message)):
                    valid_bit = np.where(message[m,:] < 1)[0]
                    if len(valid_bit) > 0:
                        # send message only when valid
                        new_message = np.concatenate((message[m,:], senders[m,:])).ravel()
                        self.last_comm[g][m*l:(m+1)*l] = new_message
                        # evaders = []
                        # for e in self._evaders[g*self.n_evaders_pergroup:(g+1)*self.n_evaders_pergroup]:
                        #     if e.visible:
                        #         evaders.append(e.position)
                        #     else:
                        #         evaders.append(np.zeros(2))
                        # self.last_comm[g] = np.concatenate(evaders)

            else:
                self.last_comm[g] = np.concatenate((message, senders), axis=1).ravel()
    
    def send_groudtruth(self):
        for g in range(self.n_groups):
            positions = []
            for e in self._evaders[g*self.n_evaders_pergroup:(g+1)*self.n_evaders_pergroup]:
                if e.visible:
                    positions.append(e.position)
                else:
                    positions.append(np.zeros(2))
            self.last_comm[g] = np.concatenate(positions)

    def convert_action(self, action):
        '''convert a discrete action to continuous acceleration'''
        action_map = np.array([[0,0], [1, 0], [0, 1], [-1, 0], [0, -1],
                            [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5],[-0.5, -0.5]])
        return action_map[action] * 0.05 # *self.pursuer_max_accel*0.5

    def step(self, action, agent_id, is_last):
        if self.dist_action:
            action = self.convert_action(action)
        else:
            action = np.asarray(action)
            action = action.reshape(2)
            speed = np.linalg.norm(action)
            if speed > self.pursuer_max_accel:
                # Limit added thrust to self.pursuer_max_accel
                action = action / speed * self.pursuer_max_accel

        p = self._pursuers[agent_id]
        if self.dist_action:
            p.set_velocity(action) # if the action is the velocity instead of acceleration
        else:
            p.set_velocity(p.velocity + action)
        
        p.set_position(p.position + self.cycle_time * p.velocity)

        # Penalize large thrusts
        accel_penalty = self.thrust_penalty * math.sqrt((action ** 2).sum())
        # Average thrust penalty among all agents, and assign each agent global portion designated by (1 - local_ratio)
        self.control_rewards = (accel_penalty / self.n_pursuers) * np.ones(self.n_pursuers) * (1 - self.local_ratio)
        # Assign the current agent the local portion designated by local_ratio
        self.control_rewards[agent_id] += accel_penalty * self.local_ratio

        if is_last:
            def move_objects(objects):
                for obj in objects:
                    # Move objects
                    obj.set_position(obj.position + self.cycle_time * obj.velocity)
                    # Bounce object if it hits a wall
                    for i in range(len(obj.position)):
                        if obj.position[i] >= 1 or obj.position[i] <= 0:
                            obj.position[i] = np.clip(obj.position[i], 0, 1)
                            obj.velocity[i] = -1 * obj.velocity[i]

            move_objects(self._evaders)
            move_objects(self._poisons)

            rewards = np.zeros(self.n_pursuers)
            sensorfeatures, collisions_pursuer_evader, collisions_pursuer_poison, rewards = self.collision_handling_subroutine(rewards, is_last)
            obs_list = self.observe_list(
                sensorfeatures, collisions_pursuer_evader, collisions_pursuer_poison)
            self.last_obs = obs_list

            # new reward: negative if there are foods that have not been eaten
            for g in range(self.n_groups):
                visibles = [e.visible for e in self._evaders[g*self.n_evaders_pergroup:(g+1)*self.n_evaders_pergroup]]
                rewards[g] -= np.array(visibles).sum() * self.food_alive_penalty

            local_reward = rewards
            global_reward = local_reward.mean()
            # Distribute local and global rewards according to local_ratio
            self.last_rewards = local_reward * self.local_ratio + global_reward * (1 - self.local_ratio)

            self.frames += 1

        if self.comm:
            return self.observe(agent_id), self.communicate(agent_id)
        else:
            return self.observe(agent_id)

    def observe(self, agent):
        return np.array(self.last_obs[agent], dtype=np.float32)

    def communicate(self, agent):
        return np.array(self.last_comm[agent], dtype=np.float32)    

    def draw_obstacles(self):
        for obstacle in self.obstacle_coords:
            assert obstacle.shape == (2,)
            x, y = obstacle
            center = (int(self.window_size * self.pixel_scale * x),
                      int(self.window_size * self.pixel_scale * y))
            color = (120, 176, 178)
            pygame.draw.circle(self.screen, color, center, self.pixel_scale * self.obstacle_radius)

    def draw_background(self):
        # -1 is building pixel flag
        color = (255, 255, 255)
        rect = pygame.Rect(0, 0, self.window_size * self.pixel_scale, self.window_size * self.pixel_scale)
        pygame.draw.rect(self.screen, color, rect)

    def draw_pursuers(self):
        for pursuer in self._pursuers:
            x, y = pursuer.position
            center = (int(self.window_size * self.pixel_scale * x),
                      int(self.window_size * self.pixel_scale * y))
            for sensor in pursuer._sensors:
                start = center
                end = center + self.pixel_scale * (pursuer._sensor_range * sensor)
                color = (0, 0, 0)
                pygame.draw.line(self.screen, color, start, end, 1)
            # color = (101, 104, 249)
            color = self.colors[pursuer._group%len(self.colors)]
            pygame.draw.circle(self.screen, color, center, self.pixel_scale * self.radius)

    def draw_evaders(self):
        for evader in self._evaders:
            if not evader.visible:
                continue
            x, y = evader.position
            center = (int(self.window_size * self.pixel_scale * x),
                      int(self.window_size * self.pixel_scale * y))
            # color = (238, 116, 106)
            color = self.colors[evader._group%len(self.colors)]

            pygame.draw.circle(self.screen, color, center, self.pixel_scale * self.radius * 2)

    def draw_poisons(self):
        for poison in self._poisons:
            x, y = poison.position
            center = (int(self.pixel_scale * x),
                      int(self.pixel_scale * y))
            # color = (145, 250, 116)
            color = (0, 0, 0)
            pygame.draw.circle(self.screen, color, center, self.pixel_scale * self.radius * self.poison_scale)

    def render(self, mode="human"):
        if not self.renderOn:
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.window_size*self.pixel_scale, self.window_size*self.pixel_scale))
            else:
                self.screen = pygame.Surface((self.window_size*self.pixel_scale, self.window_size*self.pixel_scale))
            self.renderOn = True

        self.draw_background()
        self.draw_obstacles()
        self.draw_pursuers()
        self.draw_evaders()
        self.draw_poisons()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if mode == "human":
            pygame.display.flip()
        return np.transpose(new_observation, axes=(1, 0, 2)) if mode == "rgb_array" else None
