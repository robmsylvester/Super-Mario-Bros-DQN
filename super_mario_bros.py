import logging
import os

import numpy as np

from gym import spaces
from .nes_env import NesEnv, MetaNesEnv

logger = logging.getLogger(__name__)

# (world_number, level_number, area_number, max_distance)
WORLD_NUMBER = 0
LEVEL_NUMBER = 1
AREA_NUMBER = 2
MAX_DISTANCE = 3
SMB_LEVELS = [
    (1, 1, 1, 3266), (1, 2, 2, 3266), (1, 3, 4, 2514), (1, 4, 5, 2430),
    (2, 1, 1, 3298), (2, 2, 2, 3266), (2, 3, 4, 3682), (2, 4, 5, 2430),
    (3, 1, 1, 3298), (3, 2, 2, 3442), (3, 3, 3, 2498), (3, 4, 4, 2430),
    (4, 1, 1, 3698), (4, 2, 2, 3266), (4, 3, 4, 2434), (4, 4, 5, 2942),
    (5, 1, 1, 3282), (5, 2, 2, 3298), (5, 3, 3, 2514), (5, 4, 4, 2429),
    (6, 1, 1, 3106), (6, 2, 2, 3554), (6, 3, 3, 2754), (6, 4, 4, 2429),
    (7, 1, 1, 2962), (7, 2, 2, 3266), (7, 3, 4, 3682), (7, 4, 5, 3453),
    (8, 1, 1, 6114), (8, 2, 2, 3554), (8, 3, 3, 3554), (8, 4, 4, 4989)]
SUPER_MARIO_ROM_PATH = os.path.join(os.path.dirname(__file__), 'roms', 'super-mario.nes')

# --------------
# Helper Methods
# --------------
def is_int16(str):
    try:
        int(str, 16)
        return True
    except ValueError:
        return False


# --------------
# Classes
# --------------
class SuperMarioBrosEnv(NesEnv):
    def __init__(self, draw_tiles=False, level=0):
        NesEnv.__init__(self)
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.level = level
        self.draw_tiles = 1 if draw_tiles else 0
        self._mode = 'algo'             # 'algo' or 'human'
        self.lua_path.append(os.path.join(package_directory, 'lua/super-mario-bros.lua'))
        self.tiles = None
        self.launch_vars['target'] = self._get_level_code(self.level)
        self.launch_vars['mode'] = 'algo'
        self.launch_vars['meta'] = '0'
        self.launch_vars['draw_tiles'] = str(self.draw_tiles)
        if os.path.isfile(SUPER_MARIO_ROM_PATH):
            self.rom_path = SUPER_MARIO_ROM_PATH

        self.additional_rewards = {
            "time" : -0.1, #per second that passes by
            "death" : -100., #mario dies
            "extra_life" : 100., #mario gets an extra life, which includes getting 100th coin
            "mushroom" : 20., #mario eats a mushroom to become big
            "flower" : 25., #mario eats a flower
            "mushroom_hit" : -10., #mario gets hit while big
            "flower_hit" : -15., #mario gets hit while fire mario
            "coin" : 1.0, #mario gets a coin
            "finish_level" : 100.,
            "score" : 0.1, #per 100 points
        }


        # Tile mode
        if 1 == self.draw_tiles:
            self.tile_height = 13
            self.tile_width = 16
            self.screen_height = 13
            self.screen_width = 16
            self.tiles = np.zeros(shape=(self.tile_height, self.tile_width), dtype=np.uint8)
            self.observation_space = spaces.Box(low=0, high=3, shape=(self.tile_height, self.tile_width))

    # --------------
    # Properties
    # --------------
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        self.launch_vars['mode'] = value
        self.cmd_args = ['--xscale 2', '--yscale 2', '-f 0']
        if 'human' == value:
            self.disable_out_pipe = True
            self.disable_in_pipe = True
        else:
            self.disable_out_pipe = False
            self.disable_in_pipe = False

    # --------------
    # Methods
    # --------------
    def _get_level_code(self, level):
        world_number = int(level / 4) + 1
        level_number = (level % 4) + 1
        area_number = level_number
        # Worlds 1, 2, 4, 7 have a transition as area number 2 (so 2-2 is area 3 and 3, 2-3 is area 4, 2-4 is area 5)
        if world_number in [1, 2, 4, 7] and level_number >= 2:
            area_number += 1
        return '%d%d%d' % (world_number, level_number, area_number)

    #This function has been heavily modified to make the reward involve more than just raw x distance
    #These data messages are defined in the lua file for super-mario-bros
    
    #TODO - need to edit LUA file to include star power rewards

    def _process_data_message(self, frame_number, data):
        # Format: data_<frame>#name_1:value_1|name_2:value_2|...
        if frame_number <= self.last_frame or self.info is None:
            return
        parts = data.split('|')

        #rewards have to do with x distance, coins, lifes, mushrooms and fireballs, and score. we construct them one by one
        self.reward = 0.
        if not self.episode_reward:
            self.episode_reward = 0.
        
        for part in parts:
            if part.find(':') == -1:
                continue
            parts_2 = part.split(':')
            name = parts_2[0]
            value = int(parts_2[1])



            """
            The following rewards are unimplemented, by choice
            self.additional_rewards = {
                "finish_level" : 100.
                "score" : 0.1 #per 100 points
            }
            """

            #detect if finished
            if 'is_finished' == name:
                self.is_finished = bool(value)

            #detect extra lives or deaths
            elif 'life' == name:
                #print("life change detected. life value is now %d" % value)
                self.info[name] = value
                if value < 3:
                    #print("detected death")
                    self.reward += self.additional_rewards['death']
                    #self.episode_reward += self.additional_rewards['death']
                elif value > 3:
                    #print("new life detected")
                    self.reward += self.additional_rewards['extra_life']
                    #self.episode_reward += self.additional_rewards['extra_life']


            #detect getting a coin
            elif 'coin' == name:
                #print("coin detected")
                self.reward += self.additional_rewards['coin']
                self.episode_reward += self.additional_rewards['coin']
                self.info[name] = value

            #detect if we decremented by a mario second, in which case, apply the reward
            elif 'time' == name and value < self.info[name]:
                #print("Time decrement detected")
                self.reward += self.additional_rewards['time']
                self.episode_reward += self.additional_rewards['time']
                self.info[name] = value

            #detect if distance incremented
            elif 'distance' == name:
                self.reward += value - self.info[name] #change in x distance
                self.episode_reward += value #total x distance
                self.info[name] = value #total x distance
            
            #detect if mario ate a mushroom, ate a flower, or got hit without dying
            elif 'player_status' == name:

                #2 - fire mario. only achieved if eating flower while super mario
                #1 - super mario. only achieved if eating mushroom while small mario
                #0 - small mario. only achieved if hit while super mario or fire mario. if hit while small mario, death.

                #ate a flower (assuming was still super mario. if eating flower while small mario, mario only becomes super mario so this value
                #would be a value of 1, and be caught in the value == 1 checks)
                if value == 2 and value > self.info[name]:
                    self.reward += self.additional_rewards['flower']
                    self.episode_reward += self.additional_rewards['flower']

                #if currently super mario, only need to check if this is from eating mushroom. if hit while fire mario, goes back to small mario
                elif value == 1 and value > self.info[name]:
                    self.reward += self.additional_rewards['mushroom']
                    self.episode_reward += self.additional_rewards['mushroom']

                #if small value was sent, you got hit when you were big
                elif value == 0 and self.info[name] == 1:
                    self.reward += self.additional_rewards['mushroom_hit']
                    self.episode_reward += self.additional_rewards['mushroom_hit']

                #or worse, you got hit when you were a flower
                elif value == 0 and self.info[name] == 2:
                    self.reward += self.additional_rewards['flower_hit']
                    self.episode_reward += self.additional_rewards['flower_hit']       

                self.info[name] = value

            else:
                self.info[name] = value

    def _process_screen_message(self, frame_number, data):
        # Format: screen_<frame>#<x (2 hex)><y (2 hex)><palette (2 hex)>|<x><y><p>|...
        if frame_number <= self.last_frame or self.screen is None:
            return
        parts = data.split('|')
        for part in parts:
            if 6 == len(part) and is_int16(part[0:2]) and is_int16(part[2:4]):
                x = int(part[0:2], 16)
                y = int(part[2:4], 16)
                self.screen[y][x] = self._get_rgb_from_palette(part[4:6])

    def _process_tiles_message(self, frame_number, data):
        # Format: tiles_<frame>#<x (1 hex)><y (1 hex)><value (1 hex)>|<x><y><v>|...
        if frame_number <= self.last_frame or self.tiles is None:
            return
        parts = data.split('|')
        for part in parts:
            if 3 == len(part) and is_int16(part[0:1]) and is_int16(part[1:2]) and is_int16(part[2:3]):
                x = int(part[0:1], 16)
                y = int(part[1:2], 16)
                v = int(part[2:3], 16)
                self.tiles[y][x] = v
                if v == 0: self.screen[y][x] = self._get_rgb_from_palette('0D')
                if v == 1: self.screen[y][x] = self._get_rgb_from_palette('30')
                if v == 2: self.screen[y][x] = self._get_rgb_from_palette('27')
                if v == 3: self.screen[y][x] = self._get_rgb_from_palette('05')

    def _process_ready_message(self, frame_number):
        # Format: ready_<frame>
        if 0 == self.last_frame:
            self.last_frame = frame_number

    def _process_done_message(self, frame_number):
        # Done means frame is done processing, please send next command
        # Format: done_<frame>
        if frame_number > self.last_frame:
            self.last_frame = frame_number

    def _process_reset_message(self):
        # Reset means 'changelevel' needs to be sent and last_frame needs to be set to 0
        # Not implemented in non-meta levels
        pass

    def _process_exit_message(self):
        # Exit means fceux is terminating
        # Format: exit
        self.is_finished = True
        self._is_exiting = 1
        self.close()

    def _parse_frame_number(self, parts):
        # Parsing frame number
        try:
            frame_number = int(parts[1]) if len(parts) > 1 else 0
            return frame_number
        except:
            pass

        # Sometimes beginning of message is sent twice (screen_70screen_707#)
        if len(parts) > 2 and parts[2].isdigit():
            tentative_frame = int(parts[2])
            if self.last_frame - 10 < tentative_frame < self.last_frame + 10:
                return tentative_frame

        # Otherwise trying to make sense of digits
        else:
            digits = ''.join(c for c in ''.join(parts[1:]) if c.isdigit())
            tentative_frame = int(digits) if len(digits) > 1 else 0
            if self.last_frame - 10 < tentative_frame < self.last_frame + 10:
                return tentative_frame

        # Unable to parse - Likely an invalid message
        return None

    def _process_pipe_message(self, message):
        # Parsing
        parts = message.split('#')
        header = parts[0] if len(parts) > 0 else ''
        data = parts[1] if len(parts) > 1 else ''
        parts = header.split('_')
        message_type = parts[0] if len(parts) > 0 else ''
        frame_number = self._parse_frame_number(parts)

        # Invalid message - Ignoring
        if frame_number is None:
            return

        # Processing
        if 'data' == message_type:
            self._process_data_message(frame_number, data)
        elif 'screen' == message_type:
            self._process_screen_message(frame_number, data)
        elif 'tiles' == message_type:
            self._process_tiles_message(frame_number, data)
        elif 'ready' == message_type:
            self._process_ready_message(frame_number)
        elif 'done' == message_type:
            self._process_done_message(frame_number)
        elif 'reset' == message_type:
            self._process_reset_message()
        elif 'exit' == message_type:
            self._process_exit_message()

    def _get_reward(self):
        return self.reward

    def _get_episode_reward(self):
        return self.episode_reward

    def _get_is_finished(self):
        return self.is_finished

    def _get_state(self):
        if 1 == self.draw_tiles:
            return self.tiles.copy()
        else:
            return self.screen.copy()

    def _get_info(self):
        return self.info

    def _reset_info_vars(self):
        self.info = {
            'level': self.level,
            'distance': 0,
            'score': -1,
            'coins': -1,
            'time': -1,
            'player_status': -1
        }

    #Override the step function so that it ignores total reward and episode reward
    def _step(self, action):

        # Changing level
        if self.find_new_level:
            self.change_level()

        obs, step_reward, is_finished, info = NesEnv._step(self, action)
        self.total_reward += step_reward
        self.episode_reward += step_reward
        #reward, self.total_reward = self._calculate_reward(self._get_episode_reward(), self.total_reward)
        # First step() after new episode returns the entire total reward
        # because stats_recorder resets the episode score to 0 after reset() is called
        if self.is_new_episode:
            step_reward = 0.
            #self.total_reward = 0.
            #reward = self.total_reward

        self.is_new_episode = False
        info["level"] = self.level
        info["scores"] = self.get_scores()
        info["total_reward"] = round(self.total_reward, 4)
        info["locked_levels"] = self.locked_levels

        # Indicating new level required
        if is_finished:
            self._unlock_levels()
            self.find_new_level = True

        return obs, step_reward, is_finished, info


class MetaSuperMarioBrosEnv(SuperMarioBrosEnv, MetaNesEnv):

    def __init__(self, average_over=10, passing_grade=600, min_tries_for_avg=5, draw_tiles=0):
        MetaNesEnv.__init__(self,
                            average_over=average_over,
                            passing_grade=passing_grade,
                            min_tries_for_avg=min_tries_for_avg,
                            num_levels=32)
        SuperMarioBrosEnv.__init__(self, draw_tiles=draw_tiles, level=0)
        self.launch_vars['meta'] = '1'

    def _process_reset_message(self):
        self.last_frame = 0

    def _get_standard_reward(self, episode_reward):
        #Returns a standardized reward for an episode (i.e. between 0 and 1,000)
        min_score = 0
        target_score = float(SMB_LEVELS[self.level][MAX_DISTANCE]) - 40
        max_score = min_score + (target_score - min_score) / 0.99  # Target is 99th percentile (Scale 0-1000)
        std_reward = round(1000 * (episode_reward - min_score) / (max_score - min_score), 4)
        std_reward = min(1000, std_reward)  # Cannot be more than 1,000
        std_reward = max(0, std_reward)  # Cannot be less than 0
        return std_reward

