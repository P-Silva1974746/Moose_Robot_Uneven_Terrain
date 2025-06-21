A2C reward function was altered in order to make the robot be able to learn the objective is to get to the goal
-
Reward function modifications:
        
- Reward for the robot to reach the goal was increased to:
  -     1000 + (initial distance to goal / time the robot took to get there)
- Reward in case the robot falls out of the map was decreased:
  -     -100

Termination conditions altered
-
Now the robot considers the goal reached if the ever is at less than 5m from the goal position, this tries to make the 
task easier to learn, and also make the reward of reaching the goal more present during training

Maximum distance between goal and start position
-
Now exists a maximum distance of 50m between the start position and the goal, this also tries to make the episodes shorter
possibly making the reward for reaching the goal appear more often and making the task easier to learn

New metrics introduced for measuring how close the A2C model was to reaching the goal
-
In order to more get a more precise performance analysis, new performance metrics were used to monitor the 100 test runs 
of the A2C model:
- Minimum distance to the goal during the episode
- Standard deviation of the height, of the path the robot walked

Extensive training
-
The training regime is still the same, 100 000 timesteps per map as described in the section 3F of our paper, however the
model was trained longer in order to try to reach the conversion of the mean reward rollout per episode, the model was trained 
in total for 10 million timesteps