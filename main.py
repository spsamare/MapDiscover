from init import *
from agent_definitions import Agent
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clock = pg.time.Clock()
    max_duration = 10  # in seconds
    #
    all_sprites = pg.sprite.Group()
    # agent1 = Agent()
    # all_sprites.add(agent1)
    agent_count = 0
    for i in range(NUM_AGENTS):
        all_sprites.add(
            Agent(
                id_num=agent_count,
                location=[np.random.randint(WIDTH//4, 3*WIDTH//4), np.random.randint(HEIGHT//4, 3*HEIGHT//4)],
                color=tuple(np.random.randint(128, 256, 3))
            )
        )
        agent_count += 1
    #
    done_round = False
    time_world = 0
    while not done_round:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done_round = True

        all_sprites.update()
        #
        # SCREEN.fill(COLORS["background"])
        SCREEN.blit(BACKGROUND, (0, 0))
        all_sprites.draw(SCREEN)
        pg.display.flip()
        clock.tick(FPS)
        if time_world % (FPS // 4) == 0:
            for agent in all_sprites:
                pass  # agent.snapshot(t=time_world)
                # Uncomment to save the agents' views

        time_world += 1

        if time_world >= max_duration * FPS:
            done_round = True
    """
    if time_world % FPS == 0:
        for agent in all_sprites:
            agent.snapshot(t=time_world)
    """
    pg.quit()
