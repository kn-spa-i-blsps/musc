from py_trees.behaviour import Behaviour
from py_trees.common import ParallelPolicy
from py_trees.common import Status
from py_trees.composites import Sequence, Selector, Parallel
from py_trees import common
from py_trees.decorators import Inverter


# robimy misję umijania drzewa i znajdywania czerwonej kropki
# jeśli ją znajdzie to wtedy na niej ląduje
# jeśli znajdzie drzewo to je wymija
# użyjemy drzewa behawioralnego czyli reagujące na wydarzenia dziejące się w czasie rzeczywistym

# poniżej są akcje dronów śledzących matkę

class BatteryCheck(Behaviour):
    def __init__(self, name: str):
        super(BatteryCheck, self).__init__(name)

    def initialise(self) -> None:
        pass
    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass
    def terminate(self, new_status: common.Status) -> None:
        pass

# jakiś sygnał że lądujemy
class SendSignal(Behaviour):
    def __init__(self, name: str):
        super(SendSignal, self).__init__(name)

    def initialise(self) -> None:
        pass
    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass
    def terminate(self, new_status: common.Status) -> None:
        pass


class MotherIdentification(Behaviour):
    def __init__(self, name: str):
        super(MotherIdentification, self).__init__(name)

    def initialise(self) -> None:
        pass
    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass
    def terminate(self, new_status: common.Status) -> None:
        pass

class MotherTracking(Behaviour):
    def __init__(self, name: str):
        super(MotherTracking, self).__init__(name)

    def initialise(self) -> None:
        pass
    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass
    def terminate(self, new_status: common.Status) -> None:
        pass

# akcje matki
# oraz BatteryCheck

# jeśli jest sygnał lądowania od co-dronow to wtedy lądujemy
class SignalFromCoDrones(Behaviour):
    def __init__(self, name: str):
        super(SignalFromCoDrones, self).__init__(name)

    def initialise(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass

    def terminate(self, new_status: common.Status) -> None:
        pass

class Landing(Behaviour):
    def __init__(self, name: str):
        super(Landing, self).__init__(name)

    def initialise(self) -> None:
        pass
    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass
    def terminate(self, new_status: common.Status) -> None:
        pass

class FindTree(Behaviour):
    def __init__(self, name: str):
        super(Landing, self).__init__(name)

    def initialise(self) -> None:
        pass
    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass
    def terminate(self, new_status: common.Status) -> None:
        pass

class BypassingTree(Behaviour):
    def __init__(self, name: str):
        super(BypassingTree, self).__init__(name)

    def initialise(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass

    def terminate(self, new_status: common.Status) -> None:
        pass

class GoStraight(Behaviour):
    def __init__(self, name: str):
        super(Landing, self).__init__(name)

    def initialise(self) -> None:
        pass
    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass
    def terminate(self, new_status: common.Status) -> None:
        pass

class FindRedDot(Behaviour):
    def __init__(self, name: str):
        super(FindRedDot, self).__init__(name)

    def initialise(self) -> None:
        pass
    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass
    def terminate(self, new_status: common.Status) -> None:
        pass

class FindFollowersDrones(Behaviour):
    def __init__(self, name: str):
        super(FindFollowersDrones, self).__init__(name)

    def initialise(self) -> None:
        pass
    def setup(self) -> None:
        pass

    def update(self) -> common.Status:
        pass
    def terminate(self, new_status: common.Status) -> None:
        pass


if __name__ == '__main__':

    # node wewnętrzne dla śledzących
    follower_parallel_node = Parallel(
        name="follower_parallel_node",
        policy=ParallelPolicy.SuccessOnAll() # nie ma znaczenia, bo nikomu nie przekazujemy tego wyniku dalej
    )
    follower_sequence_node = Sequence(name="follower_sequence_node", memory=True)
    follower_selector_check_battery_node = Selector(
        name="follower_selector_node",
        memory=True # w tym przypadku nie ma znacznia bo jak mam mniej niz powiedzmy 10% to nagle sie nie naładuje magicznie
    )

    # node wewnętrzne dla matki
    mother_selector_node = Selector(name="mother_selector_node", memory=True)
    mother_selector_is_road_clean_node = Selector(name="mother_selector_is_road_clean_node", memory=True)

    mother_decorator_is_road_clean_node = Inverter(
        name="mother_decorator_is_road_clean_node",
        child=mother_selector_is_road_clean_node
    )
    mother_sequence_is_road_clean_node = Sequence(name="mother_sequence_is_road_clean_node", memory=True)
    mother_sequence_check_red_dot_node = Sequence(name="mother_sequence_check_red_dot", memory=True)
    mother_sequence_check_battery_node = Sequence(name="mother_selector_check_battery_node",memory=True)
    mother_sequence_signal_from_followers_node = Sequence(name="mother_sequence_signal_from_followers_node", memory=True)

    # node zewnętrzne dla śledzących
    check_battery_follower = BatteryCheck(name="check_battery_follower")
    sending_signal = SendSignal(name="sending_signal")
    iden_mother = MotherIdentification(name="iden_mother")
    tracking_mother = MotherTracking(name="tracking_mother")


    # node zewnętrzne dla matki
    check_battery_mother = BatteryCheck(name="check_battery_mother")
    landing_1 = Landing(name="landing_1")
    receiving_signal_from_followers = SignalFromCoDrones(name="receiving_signal")
    is_road_clean = FindTree(name="finding_tree")
    bypassing = BypassingTree(name="bypassing")
    find_red_dot = FindRedDot(name="find_red_dot")
    go_straight = GoStraight(name="go_straight")
    landing_2 = Landing(name="landing_2")
    landing_3 = Landing(name="landing_3")
    find_followers = FindFollowersDrones(name="find_followers")

    # drzewko wykonuje się zawsze od lewej do prawej, więc kolejność node jest istotna
    # drzewko dla śledzących
    follower_parallel_node.add_children([
        follower_selector_check_battery_node,
        follower_sequence_node
    ])

    follower_selector_check_battery_node.add_children([
        check_battery_follower,
        sending_signal
    ])

    follower_sequence_node.add_children([
        iden_mother,
        tracking_mother
    ])

    #drzewko dla matki
    mother_selector_node.add_children([
        mother_sequence_check_battery_node,
        mother_sequence_signal_from_followers_node,
        mother_sequence_check_red_dot_node,
        mother_decorator_is_road_clean_node
    ])

    mother_sequence_check_battery_node.add_children([
        check_battery_mother,
        landing_1
    ])

    mother_sequence_signal_from_followers_node.add_children([
        find_followers,
        receiving_signal_from_followers,
        landing_2
    ])

    mother_sequence_check_red_dot_node.add_children([
        find_red_dot,
        landing_3
    ])

    mother_sequence_is_road_clean_node.add_children([
        mother_sequence_is_road_clean_node,
        go_straight
    ])

    mother_sequence_is_road_clean_node.add_children([
        is_road_clean,
        bypassing
    ])

    # misja jak to ma działać
    while True:
        mother_selector_node.tick_once()
        follower_parallel_node.tick_once()

        if mother_selector_node.status == Status.SUCCESS:
            print("Drone started landing")
            break