"""Paper-faithful VRP-RPD solver components for the Pygame demo."""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from . import config
from .model import EvaluatedSolution, Instance, Operation, OperationEvent, Routes


@dataclass
class Chromosome:
    priority_genes: List[float]
    assignment_genes: List[float]
    fitness: float | None = None
    routes: Routes | None = None


@dataclass
class SolverRunResult:
    initial: EvaluatedSolution
    alns: EvaluatedSolution
    brkga: EvaluatedSolution
    selected_label: str
    best_label: str
    best: EvaluatedSolution


@dataclass(frozen=True)
class QuickEvaluation:
    feasible: bool
    makespan: float


class VRPRPDSolver:
    """CPU implementation of the paper's ALNS -> BRKGA pipeline."""

    def __init__(
        self,
        instance: Instance,
        solver_config: config.SolverConfig | None = None,
    ) -> None:
        self.instance = instance
        self.solver_config = solver_config or config.default_runtime_config().solver
        self.speed_in_per_sec = self.instance.instance_config.alvik_speed_in_per_sec
        self.rng = random.Random(self.solver_config.random_seed)
        self.customers = list(self.instance.active_job_ids)
        self.customer_set = set(self.customers)
        self.customer_index = {
            customer_id: index for index, customer_id in enumerate(self.customers)
        }
        self.operation_gene_order = []
        self.gene_index: Dict[Tuple[int, str], int] = {}
        for customer_id in self.customers:
            for kind in ("D", "P"):
                op = Operation(customer_id, kind)
                self.gene_index[(customer_id, kind)] = len(self.operation_gene_order)
                self.operation_gene_order.append(op)

        self.node_index = {
            node_id: index
            for index, node_id in enumerate(self.instance.world.coords)
        }
        self.depot_node_index = self.node_index[self.instance.depot_node]
        self.customer_node_indices = [
            self.node_index[self.customer_node(customer_id)]
            for customer_id in self.customers
        ]
        self.processing_time_array = [
            self.instance.processing_times[customer_id]
            for customer_id in self.customers
        ]
        node_count = len(self.node_index)
        self.travel_times_dense = [
            [0.0 for _ in range(node_count)]
            for _ in range(node_count)
        ]
        for source, targets in self.instance.world.distances.items():
            source_index = self.node_index[source]
            for target, distance in targets.items():
                target_index = self.node_index[target]
                self.travel_times_dense[source_index][target_index] = (
                    distance / self.speed_in_per_sec
                )

        self.destroy_names = [
            "random",
            "worst",
            "shaw",
            "cluster",
            "route",
            "critical",
        ]
        self.repair_names = [
            "greedy",
            "regret2",
            "regret3",
            "regretm",
        ]

    def _quick_infeasible(self) -> QuickEvaluation:
        return QuickEvaluation(feasible=False, makespan=float("inf"))

    def _evaluate_cost(self, routes: Routes, require_complete: bool = True) -> QuickEvaluation:
        customer_count = len(self.customers)
        total_ops = sum(len(route) for route in routes)

        if total_ops == 0:
            if require_complete and customer_count:
                return self._quick_infeasible()
            return QuickEvaluation(feasible=True, makespan=0.0)

        drop_counts = [0] * customer_count
        pickup_counts = [0] * customer_count
        drop_indices = [-1] * customer_count
        pickup_indices = [-1] * customer_count
        adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(total_ops)]
        indegree = [0] * total_ops
        completion = [float("-inf")] * total_ops
        source_edges: List[Tuple[int, float]] = []
        last_op = [-1] * self.instance.vehicle_count
        last_station = [self.depot_node_index] * self.instance.vehicle_count

        customer_index = self.customer_index
        customer_node_indices = self.customer_node_indices
        travel_times = self.travel_times_dense
        capacity = self.instance.capacity

        global_op_index = 0
        for vehicle_id, route in enumerate(routes):
            load = capacity
            prev_op = -1
            prev_station = self.depot_node_index
            for operation in route:
                if operation.kind not in {"D", "P"}:
                    return self._quick_infeasible()
                customer_idx = customer_index[operation.customer_id]
                station_idx = customer_node_indices[customer_idx]
                is_pickup = operation.kind == "P"
                if is_pickup:
                    pickup_counts[customer_idx] += 1
                    pickup_indices[customer_idx] = global_op_index
                    load += 1
                else:
                    drop_counts[customer_idx] += 1
                    drop_indices[customer_idx] = global_op_index
                    load -= 1
                if load < 0 or load > capacity:
                    return self._quick_infeasible()

                travel = travel_times[prev_station][station_idx]
                if prev_op == -1:
                    source_edges.append((global_op_index, travel))
                else:
                    adjacency[prev_op].append((global_op_index, travel))
                    indegree[global_op_index] += 1
                prev_op = global_op_index
                prev_station = station_idx
                global_op_index += 1

            if prev_op != -1:
                last_op[vehicle_id] = prev_op
                last_station[vehicle_id] = prev_station

        for customer_idx in range(customer_count):
            drops = drop_counts[customer_idx]
            pickups = pickup_counts[customer_idx]
            if drops == pickups == 0:
                if require_complete:
                    return self._quick_infeasible()
                continue
            if drops != 1 or pickups != 1:
                return self._quick_infeasible()
            drop_op = drop_indices[customer_idx]
            pickup_op = pickup_indices[customer_idx]
            adjacency[drop_op].append((pickup_op, self.processing_time_array[customer_idx]))
            indegree[pickup_op] += 1

        for op_index, travel in source_edges:
            completion[op_index] = max(completion[op_index], travel)

        queue = deque(
            op_index
            for op_index in range(total_ops)
            if indegree[op_index] == 0
        )
        processed = 0
        while queue:
            node = queue.popleft()
            processed += 1
            node_time = completion[node]
            for neighbor, weight in adjacency[node]:
                candidate = node_time + weight
                if candidate > completion[neighbor]:
                    completion[neighbor] = candidate
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if processed != total_ops:
            return self._quick_infeasible()

        makespan = 0.0
        for vehicle_id, op_index in enumerate(last_op):
            if op_index < 0:
                continue
            return_time = completion[op_index] + travel_times[last_station[vehicle_id]][self.depot_node_index]
            if return_time > makespan:
                makespan = return_time

        return QuickEvaluation(feasible=True, makespan=makespan)

    def solve(self) -> SolverRunResult:
        initial_routes = self.construct_initial_solution()
        initial_eval = self.evaluate(initial_routes, require_complete=True)

        alns_routes, alns_eval = self.run_alns(initial_routes)
        brkga_routes, brkga_eval = self.run_brkga(alns_routes)

        labeled = {
            "initial": initial_eval,
            "alns": alns_eval,
            "brkga": brkga_eval,
        }
        best_label = min(labeled, key=lambda label: labeled[label].makespan)
        return SolverRunResult(
            initial=initial_eval,
            alns=alns_eval,
            brkga=brkga_eval,
            selected_label="paper",
            best_label=best_label,
            best=labeled[best_label],
        )

    # ------------------------------------------------------------------
    # Exact route evaluation
    # ------------------------------------------------------------------

    def evaluate(self, routes: Routes, require_complete: bool = True) -> EvaluatedSolution:
        routes = clone_routes(routes)
        customer_counts = {
            customer_id: {"D": 0, "P": 0}
            for customer_id in self.customers
        }

        node_for_vehicle_op: Dict[Tuple[int, int], str] = {}
        drop_node_for_customer: Dict[int, str] = {}
        pickup_node_for_customer: Dict[int, str] = {}
        load_before_after: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for vehicle_id, route in enumerate(routes):
            load = self.instance.capacity
            for op_index, operation in enumerate(route):
                if operation.kind not in {"D", "P"}:
                    return self._infeasible(routes, f"Unknown operation type: {operation.kind}")
                customer_counts[operation.customer_id][operation.kind] += 1
                node_id = f"v{vehicle_id}_op{op_index}_{operation.kind}{operation.customer_id}"
                node_for_vehicle_op[(vehicle_id, op_index)] = node_id
                before = load
                load += -1 if operation.kind == "D" else 1
                if load < 0 or load > self.instance.capacity:
                    return self._infeasible(
                        routes,
                        f"Capacity violation on vehicle {vehicle_id} at {operation.key}",
                    )
                load_before_after[(vehicle_id, op_index)] = (before, load)

        present_customers = set()
        for customer_id, counts in customer_counts.items():
            if counts["D"] == counts["P"] == 0:
                if require_complete:
                    return self._infeasible(
                        routes,
                        f"Missing customer {customer_id} in complete evaluation",
                    )
                continue
            if counts["D"] != 1 or counts["P"] != 1:
                return self._infeasible(routes, f"Unpaired operations for customer {customer_id}")
            present_customers.add(customer_id)

        adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        indegree: Dict[str, int] = defaultdict(int)
        nodes = {"source"}
        route_predecessor: Dict[str, str] = {}
        route_edge_weight: Dict[str, float] = {}
        route_source_location: Dict[str, str] = {}
        last_node_for_vehicle: Dict[int, str | None] = {}
        last_loc_for_vehicle: Dict[int, str] = {}

        for vehicle_id, route in enumerate(routes):
            prev_node = "source"
            prev_loc = self.instance.depot_node
            for op_index, operation in enumerate(route):
                node_id = node_for_vehicle_op[(vehicle_id, op_index)]
                station_node = self.customer_node(operation.customer_id)
                travel = self.travel_time(prev_loc, station_node)
                nodes.add(node_id)
                adjacency[prev_node].append((node_id, travel))
                indegree[node_id] += 1
                route_predecessor[node_id] = prev_node
                route_edge_weight[node_id] = travel
                route_source_location[node_id] = prev_loc
                if operation.kind == "D":
                    drop_node_for_customer[operation.customer_id] = node_id
                else:
                    pickup_node_for_customer[operation.customer_id] = node_id
                prev_node = node_id
                prev_loc = station_node
            last_node_for_vehicle[vehicle_id] = None if not route else prev_node
            last_loc_for_vehicle[vehicle_id] = prev_loc

        for customer_id in present_customers:
            drop_node = drop_node_for_customer[customer_id]
            pickup_node = pickup_node_for_customer[customer_id]
            processing = self.instance.processing_times[customer_id]
            adjacency[drop_node].append((pickup_node, processing))
            indegree[pickup_node] += 1

        indegree.setdefault("source", 0)
        topo_queue = deque(node for node in nodes if indegree[node] == 0)
        topo_order: List[str] = []
        completion_times: Dict[str, float] = {"source": 0.0}
        while topo_queue:
            node = topo_queue.popleft()
            topo_order.append(node)
            node_time = completion_times.get(node, float("-inf"))
            for neighbor, weight in adjacency.get(node, []):
                candidate = node_time + weight
                if candidate > completion_times.get(neighbor, float("-inf")):
                    completion_times[neighbor] = candidate
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    topo_queue.append(neighbor)

        if len(topo_order) != len(nodes):
            return self._infeasible(routes, "Precedence cycle detected")

        events_by_vehicle: Dict[int, List[OperationEvent]] = defaultdict(list)
        drop_times: Dict[int, float] = {}
        pickup_times: Dict[int, float] = {}
        drop_vehicle: Dict[int, int] = {}
        pickup_vehicle: Dict[int, int] = {}
        return_times: Dict[int, float] = {}

        for vehicle_id, route in enumerate(routes):
            prev_completion = 0.0
            prev_loc = self.instance.depot_node
            for op_index, operation in enumerate(route):
                node_id = node_for_vehicle_op[(vehicle_id, op_index)]
                station_node = self.customer_node(operation.customer_id)
                arrival = prev_completion + self.travel_time(prev_loc, station_node)
                completion = completion_times[node_id]
                wait_time = max(0.0, completion - arrival)
                load_before, load_after = load_before_after[(vehicle_id, op_index)]
                event = OperationEvent(
                    vehicle_id=vehicle_id,
                    op_index=op_index,
                    operation=operation,
                    node_id=station_node,
                    arrival_time=arrival,
                    completion_time=completion,
                    wait_time=wait_time,
                    load_before=load_before,
                    load_after=load_after,
                )
                events_by_vehicle[vehicle_id].append(event)
                if operation.kind == "D":
                    drop_times[operation.customer_id] = completion
                    drop_vehicle[operation.customer_id] = vehicle_id
                else:
                    pickup_times[operation.customer_id] = completion
                    pickup_vehicle[operation.customer_id] = vehicle_id
                prev_completion = completion
                prev_loc = station_node

            last_node = last_node_for_vehicle[vehicle_id]
            if last_node is None:
                return_times[vehicle_id] = 0.0
            else:
                return_times[vehicle_id] = (
                    completion_times[last_node]
                    + self.travel_time(last_loc_for_vehicle[vehicle_id], self.instance.depot_node)
                )

        makespan = max(return_times.values(), default=0.0)
        critical_vehicle = max(return_times, key=return_times.get, default=0)
        return EvaluatedSolution(
            routes=routes,
            feasible=True,
            makespan=makespan,
            events_by_vehicle=dict(events_by_vehicle),
            drop_times=drop_times,
            pickup_times=pickup_times,
            return_times=return_times,
            drop_vehicle=drop_vehicle,
            pickup_vehicle=pickup_vehicle,
            critical_vehicle=critical_vehicle,
            completion_times=completion_times,
        )

    def _infeasible(self, routes: Routes, reason: str) -> EvaluatedSolution:
        return EvaluatedSolution(
            routes=clone_routes(routes),
            feasible=False,
            makespan=float("inf"),
            events_by_vehicle={},
            drop_times={},
            pickup_times={},
            return_times={},
            drop_vehicle={},
            pickup_vehicle={},
            critical_vehicle=0,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Initial solution
    # ------------------------------------------------------------------

    def construct_initial_solution(self) -> Routes:
        clusters = self._sweep_clusters()
        clusters = self._balance_clusters(clusters)
        routes: Routes = [[] for _ in range(self.instance.vehicle_count)]
        for vehicle_id, cluster in enumerate(clusters):
            routes[vehicle_id] = self._build_cluster_route(cluster)

        current = routes
        for _ in range(5):
            improved = False
            current, changed = self._relocate_customer_pass(current, max_passes=1)
            improved |= changed
            current, changed = self._pickup_reposition_pass(current, max_passes=1)
            improved |= changed
            if not improved:
                break
        return current

    def _sweep_clusters(self) -> List[List[int]]:
        depot_x, depot_y = self.instance.world.depot_anchor
        ordered = sorted(
            self.customers,
            key=lambda customer_id: math.atan2(
                self.instance.station_by_id[customer_id].coord[1] - depot_y,
                self.instance.station_by_id[customer_id].coord[0] - depot_x,
            ),
        )
        clusters = [[] for _ in range(self.instance.vehicle_count)]
        base_size = len(ordered) // self.instance.vehicle_count
        extras = len(ordered) % self.instance.vehicle_count
        cursor = 0
        for vehicle_id in range(self.instance.vehicle_count):
            take = base_size + (1 if vehicle_id < extras else 0)
            clusters[vehicle_id] = ordered[cursor : cursor + take]
            cursor += take
        return clusters

    def _balance_clusters(self, clusters: List[List[int]]) -> List[List[int]]:
        clusters = [list(cluster) for cluster in clusters]
        if not any(clusters):
            return clusters

        for _ in range(60):
            workloads = [sum(self.customer_workload(customer_id) for customer_id in cluster) for cluster in clusters]
            avg = sum(workloads) / len(workloads)
            max_idx = max(range(len(clusters)), key=lambda idx: workloads[idx])
            min_idx = min(range(len(clusters)), key=lambda idx: workloads[idx])
            if workloads[max_idx] <= avg * 1.10:
                break
            if not clusters[max_idx]:
                break

            best_customer = None
            best_score = float("inf")
            for customer_id in clusters[max_idx]:
                new_over = workloads[max_idx] - self.customer_workload(customer_id)
                new_under = workloads[min_idx] + self.customer_workload(customer_id)
                score = abs(new_over - avg) + abs(new_under - avg)
                if score < best_score:
                    best_score = score
                    best_customer = customer_id
            if best_customer is None:
                break
            clusters[max_idx].remove(best_customer)
            clusters[min_idx].append(best_customer)

        return clusters

    def _build_cluster_route(self, cluster: Sequence[int]) -> List[Operation]:
        pending = set(cluster)
        active: Dict[int, float] = {}
        route: List[Operation] = []
        current_node = self.instance.depot_node
        current_time = 0.0
        load = self.instance.capacity

        while pending or active:
            candidates = []
            if load > 0:
                for customer_id in pending:
                    station_node = self.customer_node(customer_id)
                    travel = self.travel_time(current_node, station_node)
                    completion = current_time + travel
                    candidates.append((completion, travel, Operation(customer_id, "D")))
            if load < self.instance.capacity:
                for customer_id, drop_time in active.items():
                    station_node = self.customer_node(customer_id)
                    travel = self.travel_time(current_node, station_node)
                    arrival = current_time + travel
                    completion = max(arrival, drop_time + self.instance.processing_times[customer_id])
                    candidates.append((completion, travel, Operation(customer_id, "P")))

            if not candidates:
                break

            _, _, chosen = min(
                candidates,
                key=lambda item: (item[0], item[1], item[2].customer_id, item[2].kind),
            )
            station_node = self.customer_node(chosen.customer_id)
            arrival = current_time + self.travel_time(current_node, station_node)
            route.append(chosen)
            if chosen.kind == "D":
                current_time = arrival
                pending.remove(chosen.customer_id)
                active[chosen.customer_id] = current_time
                load -= 1
            else:
                current_time = max(
                    arrival,
                    active[chosen.customer_id] + self.instance.processing_times[chosen.customer_id],
                )
                active.pop(chosen.customer_id)
                load += 1
            current_node = station_node

        return route

    # ------------------------------------------------------------------
    # ALNS
    # ------------------------------------------------------------------

    def run_alns(self, initial_routes: Routes) -> Tuple[Routes, EvaluatedSolution]:
        current_routes = clone_routes(initial_routes)
        current_eval = self.evaluate(current_routes, require_complete=True)
        best_routes = clone_routes(current_routes)
        best_eval = current_eval

        destroy_weights = {name: 1.0 for name in self.destroy_names}
        repair_weights = {name: 1.0 for name in self.repair_names}
        destroy_scores = {name: 0.0 for name in self.destroy_names}
        repair_scores = {name: 0.0 for name in self.repair_names}
        destroy_attempts = {name: 0 for name in self.destroy_names}
        repair_attempts = {name: 0 for name in self.repair_names}

        temperature_init = self.solver_config.alns_initial_temperature_coeff * best_eval.makespan
        temperature = temperature_init
        stagnation = 0

        for iteration in range(1, self.solver_config.alns_max_iterations + 1):
            destroy_name = self._weighted_choice(destroy_weights)
            repair_name = self._weighted_choice(repair_weights)
            destroy_attempts[destroy_name] += 1
            repair_attempts[repair_name] += 1

            q = self._sample_removal_size()
            partial_routes, removed_customers = self._destroy(current_routes, current_eval, destroy_name, q)
            candidate_routes = self._repair(partial_routes, removed_customers, repair_name)
            quick_candidate = self._evaluate_cost(candidate_routes, require_complete=True)
            if not quick_candidate.feasible:
                candidate_eval = self._fallback_repair(partial_routes, removed_customers)
                candidate_routes = candidate_eval.routes
                candidate_makespan = candidate_eval.makespan
            else:
                candidate_eval = None
                candidate_makespan = quick_candidate.makespan

            accepted = False
            if candidate_makespan < current_eval.makespan:
                if candidate_eval is None:
                    candidate_eval = self.evaluate(candidate_routes, require_complete=True)
                current_routes = candidate_routes
                current_eval = candidate_eval
                destroy_scores[destroy_name] += self.solver_config.alns_score_sigma_2
                repair_scores[repair_name] += self.solver_config.alns_score_sigma_2
                accepted = True
                if candidate_makespan < best_eval.makespan:
                    best_routes = clone_routes(candidate_routes)
                    best_eval = candidate_eval
                    destroy_scores[destroy_name] += (
                        self.solver_config.alns_score_sigma_1
                        - self.solver_config.alns_score_sigma_2
                    )
                    repair_scores[repair_name] += (
                        self.solver_config.alns_score_sigma_1
                        - self.solver_config.alns_score_sigma_2
                    )
                    stagnation = 0
            else:
                delta = candidate_makespan - current_eval.makespan
                if temperature > 0 and self.rng.random() < math.exp(-(delta / temperature)):
                    if candidate_eval is None:
                        candidate_eval = self.evaluate(candidate_routes, require_complete=True)
                    current_routes = candidate_routes
                    current_eval = candidate_eval
                    destroy_scores[destroy_name] += self.solver_config.alns_score_sigma_3
                    repair_scores[repair_name] += self.solver_config.alns_score_sigma_3
                    accepted = True

            temperature *= self.solver_config.alns_cooling_rate
            if accepted and candidate_eval.makespan < best_eval.makespan:
                stagnation = 0
            else:
                stagnation += 1

            if (
                stagnation >= self.solver_config.alns_stagnation_threshold
                and temperature < (0.01 * temperature_init)
            ):
                temperature = self.solver_config.alns_reheat_factor * temperature_init
                stagnation = 0

            if iteration % self.solver_config.alns_pickup_reposition_interval == 0:
                current_routes, changed = self._pickup_reposition_pass(current_routes, max_passes=3)
                if changed:
                    current_eval = self.evaluate(current_routes, require_complete=True)
                    if current_eval.makespan < best_eval.makespan:
                        best_routes = clone_routes(current_routes)
                        best_eval = current_eval

            if iteration % self.solver_config.alns_cross_agent_relocate_interval == 0:
                current_routes, changed = self._cross_agent_relocation_pass(current_routes, max_passes=3)
                if changed:
                    current_eval = self.evaluate(current_routes, require_complete=True)
                    if current_eval.makespan < best_eval.makespan:
                        best_routes = clone_routes(current_routes)
                        best_eval = current_eval

            if iteration % self.solver_config.alns_weight_update_interval == 0:
                for name in self.destroy_names:
                    if destroy_attempts[name]:
                        destroy_weights[name] = max(
                            self.solver_config.alns_min_operator_weight,
                            (
                                (1.0 - self.solver_config.alns_reaction_factor)
                                * destroy_weights[name]
                            )
                            + (
                                self.solver_config.alns_reaction_factor
                                * (destroy_scores[name] / destroy_attempts[name])
                            ),
                        )
                    destroy_scores[name] = 0.0
                    destroy_attempts[name] = 0
                for name in self.repair_names:
                    if repair_attempts[name]:
                        repair_weights[name] = max(
                            self.solver_config.alns_min_operator_weight,
                            (
                                (1.0 - self.solver_config.alns_reaction_factor)
                                * repair_weights[name]
                            )
                            + (
                                self.solver_config.alns_reaction_factor
                                * (repair_scores[name] / repair_attempts[name])
                            ),
                        )
                    repair_scores[name] = 0.0
                    repair_attempts[name] = 0

        return best_routes, best_eval

    def _sample_removal_size(self) -> int:
        n = len(self.customers)
        q_max = max(4, math.floor(0.05 * n))
        q_min = max(4, math.floor(q_max / 2))
        return self.rng.randint(q_min, q_max)

    def _destroy(
        self,
        routes: Routes,
        evaluation: EvaluatedSolution,
        destroy_name: str,
        q: int,
    ) -> Tuple[Routes, List[int]]:
        if destroy_name == "random":
            removed = self.rng.sample(self.customers, q)
        elif destroy_name == "worst":
            removed = self._destroy_worst(routes, evaluation, q)
        elif destroy_name == "shaw":
            removed = self._destroy_shaw(routes, evaluation, q)
        elif destroy_name == "cluster":
            removed = self._destroy_cluster(q)
        elif destroy_name == "route":
            removed = self._destroy_route(routes, q)
        elif destroy_name == "critical":
            removed = self._destroy_critical(routes, evaluation, q)
        else:
            raise ValueError(f"Unknown destroy operator: {destroy_name}")

        partial = clone_routes(routes)
        for customer_id in removed:
            partial = remove_customer(partial, customer_id)
        return partial, removed

    def _destroy_worst(
        self,
        routes: Routes,
        evaluation: EvaluatedSolution,
        q: int,
    ) -> List[int]:
        gains = []
        for customer_id in self.customers:
            candidate = remove_customer(routes, customer_id)
            candidate_eval = self._evaluate_cost(candidate, require_complete=False)
            gain = evaluation.makespan - candidate_eval.makespan
            gains.append((customer_id, gain))
        gains.sort(key=lambda item: item[1], reverse=True)
        ranked = [customer_id for customer_id, _ in gains]
        removed = []
        while ranked and len(removed) < q:
            weights = [
            1.0 / ((rank + 1) ** self.solver_config.alns_worst_removal_power)
                for rank in range(len(ranked))
            ]
            chosen = self.rng.choices(ranked, weights=weights, k=1)[0]
            removed.append(chosen)
            ranked.remove(chosen)
        return removed

    def _destroy_shaw(
        self,
        routes: Routes,
        evaluation: EvaluatedSolution,
        q: int,
    ) -> List[int]:
        seed = self.rng.choice(self.customers)
        removed = [seed]
        while len(removed) < q:
            best_customer = None
            best_relatedness = float("inf")
            for customer_id in self.customers:
                if customer_id in removed:
                    continue
                relatedness = min(
                    self._shaw_relatedness(customer_id, other, evaluation)
                    for other in removed
                )
                if relatedness < best_relatedness:
                    best_relatedness = relatedness
                    best_customer = customer_id
            if best_customer is None:
                break
            removed.append(best_customer)
        return removed

    def _destroy_cluster(self, q: int) -> List[int]:
        center = self.rng.choice(self.customers)
        ranked = sorted(
            self.customers,
            key=lambda customer_id: (
                self.travel_time(self.customer_node(center), self.customer_node(customer_id)),
                customer_id,
            ),
        )
        return ranked[:q]

    def _destroy_route(self, routes: Routes, q: int) -> List[int]:
        route_indices = [idx for idx, route in enumerate(routes) if route]
        if not route_indices:
            return []
        selected_vehicle = self.rng.choice(route_indices)
        removed = sorted(vehicle_customers(routes[selected_vehicle]))
        if len(removed) >= q:
            return removed
        seed_nodes = [self.customer_node(customer_id) for customer_id in removed]
        candidates = [customer_id for customer_id in self.customers if customer_id not in removed]
        candidates.sort(
            key=lambda customer_id: min(
                self.travel_time(self.customer_node(customer_id), seed_node)
                for seed_node in seed_nodes
            )
        )
        removed.extend(candidates[: max(0, q - len(removed))])
        return removed

    def _destroy_critical(
        self,
        routes: Routes,
        evaluation: EvaluatedSolution,
        q: int,
    ) -> List[int]:
        critical_vehicle = evaluation.critical_vehicle
        candidates = list(vehicle_customers(routes[critical_vehicle]))
        if not candidates:
            return self.rng.sample(self.customers, q)
        gains = []
        for customer_id in candidates:
            candidate = remove_customer(routes, customer_id)
            candidate_eval = self._evaluate_cost(candidate, require_complete=False)
            gains.append((customer_id, evaluation.makespan - candidate_eval.makespan))
        gains.sort(key=lambda item: item[1], reverse=True)
        removed = [customer_id for customer_id, _ in gains[:q]]
        if len(removed) < q:
            remaining = [customer_id for customer_id in self.customers if customer_id not in removed]
            remaining.sort(key=lambda customer_id: self.customer_workload(customer_id), reverse=True)
            removed.extend(remaining[: q - len(removed)])
        return removed

    def _shaw_relatedness(
        self,
        customer_a: int,
        customer_b: int,
        evaluation: EvaluatedSolution,
    ) -> float:
        distance = self.travel_time(self.customer_node(customer_a), self.customer_node(customer_b))
        time_gap = abs(
            evaluation.drop_times.get(customer_a, 0.0) - evaluation.drop_times.get(customer_b, 0.0)
        )
        agent_mismatch = 0.0
        if evaluation.drop_vehicle.get(customer_a) != evaluation.drop_vehicle.get(customer_b):
                    agent_mismatch += self.solver_config.alns_shaw_omega / 2.0
        if evaluation.pickup_vehicle.get(customer_a) != evaluation.pickup_vehicle.get(customer_b):
                    agent_mismatch += self.solver_config.alns_shaw_omega / 2.0
        return (
                    (self.solver_config.alns_shaw_phi * distance)
                    + (self.solver_config.alns_shaw_chi * time_gap)
            + agent_mismatch
        )

    # ------------------------------------------------------------------
    # Repair operators
    # ------------------------------------------------------------------

    def _repair(self, partial_routes: Routes, removed_customers: Sequence[int], repair_name: str) -> Routes:
        if repair_name == "greedy":
            return self._repair_greedy(partial_routes, removed_customers)
        if repair_name == "regret2":
            return self._repair_regret(partial_routes, removed_customers, top_k=2)
        if repair_name == "regret3":
            return self._repair_regret(partial_routes, removed_customers, top_k=3)
        if repair_name == "regretm":
            return self._repair_regret(
                partial_routes,
                removed_customers,
                top_k=max(3, self.instance.vehicle_count),
            )
        raise ValueError(f"Unknown repair operator: {repair_name}")

    def _repair_greedy(self, partial_routes: Routes, removed_customers: Sequence[int]) -> Routes:
        current = clone_routes(partial_routes)
        remaining = set(removed_customers)
        while remaining:
            best_customer = None
            best_routes = None
            best_cost = float("inf")
            for customer_id in remaining:
                candidates = self._enumerate_customer_insertions(current, customer_id, top_k=1)
                if not candidates:
                    continue
                candidate_cost, candidate_routes = candidates[0]
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_customer = customer_id
                    best_routes = candidate_routes
            if best_customer is None or best_routes is None:
                return current
            current = best_routes
            remaining.remove(best_customer)
        return current

    def _repair_regret(
        self,
        partial_routes: Routes,
        removed_customers: Sequence[int],
        top_k: int,
    ) -> Routes:
        current = clone_routes(partial_routes)
        remaining = set(removed_customers)
        while remaining:
            selected_customer = None
            selected_routes = None
            selected_best_cost = float("inf")
            selected_regret = float("-inf")
            for customer_id in remaining:
                candidates = self._enumerate_customer_insertions(current, customer_id, top_k=top_k)
                if not candidates:
                    continue
                best_cost, best_routes = candidates[0]
                reference_costs = [cost for cost, _ in candidates[:top_k]]
                while len(reference_costs) < top_k:
                    reference_costs.append(reference_costs[-1])
                regret = sum(cost - reference_costs[0] for cost in reference_costs[1:])
                if (
                    regret > selected_regret
                    or (math.isclose(regret, selected_regret) and best_cost < selected_best_cost)
                ):
                    selected_customer = customer_id
                    selected_routes = best_routes
                    selected_best_cost = best_cost
                    selected_regret = regret
            if selected_customer is None or selected_routes is None:
                return current
            current = selected_routes
            remaining.remove(selected_customer)
        return current

    def _fallback_repair(self, partial_routes: Routes, removed_customers: Sequence[int]) -> EvaluatedSolution:
        repaired = clone_routes(partial_routes)
        for customer_id in removed_customers:
            candidates = self._enumerate_customer_insertions(repaired, customer_id, top_k=1)
            if not candidates:
                continue
            _, repaired = candidates[0]
        return self.evaluate(repaired, require_complete=True)

    def _enumerate_customer_insertions(
        self,
        routes: Routes,
        customer_id: int,
        top_k: int,
    ) -> List[Tuple[float, Routes]]:
        options: List[Tuple[float, Routes]] = []
        base = remove_customer(routes, customer_id)
        for drop_vehicle in range(self.instance.vehicle_count):
            for drop_pos in range(len(base[drop_vehicle]) + 1):
                with_drop = clone_routes(base)
                with_drop[drop_vehicle].insert(drop_pos, Operation(customer_id, "D"))
                for pickup_vehicle in range(self.instance.vehicle_count):
                    pickup_start = 0
                    if pickup_vehicle == drop_vehicle:
                        pickup_start = drop_pos + 1
                    for pickup_pos in range(pickup_start, len(with_drop[pickup_vehicle]) + 1):
                        candidate = clone_routes(with_drop)
                        candidate[pickup_vehicle].insert(pickup_pos, Operation(customer_id, "P"))
                        candidate_eval = self._evaluate_cost(candidate, require_complete=False)
                        if candidate_eval.feasible:
                            options.append((candidate_eval.makespan, candidate))
        options.sort(key=lambda item: item[0])
        return options[:top_k]

    def _enumerate_pickup_insertions(
        self,
        routes: Routes,
        customer_id: int,
    ) -> List[Tuple[float, Routes]]:
        options: List[Tuple[float, Routes]] = []
        base = remove_pickup(routes, customer_id)
        for pickup_vehicle in range(self.instance.vehicle_count):
            for pickup_pos in range(len(base[pickup_vehicle]) + 1):
                candidate = clone_routes(base)
                candidate[pickup_vehicle].insert(pickup_pos, Operation(customer_id, "P"))
                candidate_eval = self._evaluate_cost(candidate, require_complete=True)
                if candidate_eval.feasible:
                    options.append((candidate_eval.makespan, candidate))
        options.sort(key=lambda item: item[0])
        return options

    # ------------------------------------------------------------------
    # Periodic local search
    # ------------------------------------------------------------------

    def _pickup_reposition_pass(self, routes: Routes, max_passes: int) -> Tuple[Routes, bool]:
        current = clone_routes(routes)
        changed = False
        for _ in range(max_passes):
            current_eval = self.evaluate(current, require_complete=True)
            threshold = current_eval.makespan * 0.90
            candidate_vehicles = {
                vehicle_id
                for vehicle_id, return_time in current_eval.return_times.items()
                if return_time >= threshold
            }
            pickup_customers = [
                customer_id
                for customer_id, vehicle_id in current_eval.pickup_vehicle.items()
                if vehicle_id in candidate_vehicles
            ]
            best_option = None
            for customer_id in pickup_customers:
                options = self._enumerate_pickup_insertions(current, customer_id)
                if not options:
                    continue
                best_cost, candidate_routes = options[0]
                if best_cost < current_eval.makespan:
                    if best_option is None or best_cost < best_option[0]:
                        best_option = (best_cost, candidate_routes)
            if best_option is None:
                break
            _, current = best_option
            changed = True
        return current, changed

    def _relocate_customer_pass(self, routes: Routes, max_passes: int) -> Tuple[Routes, bool]:
        current = clone_routes(routes)
        changed = False
        for _ in range(max_passes):
            current_eval = self.evaluate(current, require_complete=True)
            best_option = None
            for customer_id in self.customers:
                options = self._enumerate_customer_insertions(current, customer_id, top_k=1)
                if not options:
                    continue
                best_cost, candidate_routes = options[0]
                if best_cost < current_eval.makespan:
                    if best_option is None or best_cost < best_option[0]:
                        best_option = (best_cost, candidate_routes)
            if best_option is None:
                break
            _, current = best_option
            changed = True
        return current, changed

    def _cross_agent_relocation_pass(self, routes: Routes, max_passes: int) -> Tuple[Routes, bool]:
        current = clone_routes(routes)
        changed = False
        for _ in range(max_passes):
            current_eval = self.evaluate(current, require_complete=True)
            critical_vehicle = current_eval.critical_vehicle
            critical_customers = sorted(
                customers_with_operation_on_vehicle(current, critical_vehicle)
            )
            best_option = None
            for customer_id in critical_customers:
                base = remove_customer(current, customer_id)
                for drop_vehicle in range(self.instance.vehicle_count):
                    if drop_vehicle == critical_vehicle:
                        continue
                    for drop_pos in range(len(base[drop_vehicle]) + 1):
                        with_drop = clone_routes(base)
                        with_drop[drop_vehicle].insert(drop_pos, Operation(customer_id, "D"))
                        for pickup_vehicle in range(self.instance.vehicle_count):
                            if pickup_vehicle == critical_vehicle:
                                continue
                            pickup_start = 0
                            if pickup_vehicle == drop_vehicle:
                                pickup_start = drop_pos + 1
                            for pickup_pos in range(
                                pickup_start, len(with_drop[pickup_vehicle]) + 1
                            ):
                                candidate = clone_routes(with_drop)
                                candidate[pickup_vehicle].insert(
                                    pickup_pos, Operation(customer_id, "P")
                                )
                                candidate_eval = self._evaluate_cost(candidate, require_complete=True)
                                if not candidate_eval.feasible:
                                    continue
                                if candidate_eval.makespan < current_eval.makespan:
                                    if best_option is None or candidate_eval.makespan < best_option[0]:
                                        best_option = (candidate_eval.makespan, candidate)
            if best_option is None:
                break
            _, current = best_option
            changed = True
        return current, changed

    # ------------------------------------------------------------------
    # BRKGA
    # ------------------------------------------------------------------

    def run_brkga(self, alns_routes: Routes) -> Tuple[Routes, EvaluatedSolution]:
        population = self._build_initial_population(alns_routes)
        elite_count = max(
            1,
            int(
                self.solver_config.brkga_elite_proportion
                * self.solver_config.brkga_population_size
            ),
        )
        mutant_count = max(
            1,
            int(
                self.solver_config.brkga_mutant_proportion
                * self.solver_config.brkga_population_size
            ),
        )

        best_chromosome = None
        best_eval = None
        for _ in range(self.solver_config.brkga_generations):
            for chromosome in population:
                if chromosome.fitness is None:
                    chromosome.fitness, chromosome.routes = self._decode_chromosome(chromosome)

            population.sort(key=lambda chromosome: chromosome.fitness or float("inf"))
            if best_chromosome is None or (population[0].fitness or float("inf")) < (
                best_chromosome.fitness or float("inf")
            ):
                best_chromosome = population[0]
                best_eval = self.evaluate(best_chromosome.routes or [], require_complete=True)

            elites = population[:elite_count]
            non_elites = population[elite_count:]
            next_population = [
                Chromosome(
                    priority_genes=list(chrom.priority_genes),
                    assignment_genes=list(chrom.assignment_genes),
                    fitness=chrom.fitness,
                    routes=clone_routes(chrom.routes or []),
                )
                for chrom in elites
            ]

            for _ in range(mutant_count):
                next_population.append(self._random_chromosome())

            while len(next_population) < self.solver_config.brkga_population_size:
                elite_parent = self.rng.choice(elites)
                non_elite_parent = self.rng.choice(non_elites)
                child_priority = []
                child_assignment = []
                for elite_gene, non_elite_gene in zip(
                    elite_parent.priority_genes, non_elite_parent.priority_genes
                ):
                    if self.rng.random() < self.solver_config.brkga_elite_bias:
                        child_priority.append(elite_gene)
                    else:
                        child_priority.append(non_elite_gene)
                for elite_gene, non_elite_gene in zip(
                    elite_parent.assignment_genes, non_elite_parent.assignment_genes
                ):
                    if self.rng.random() < self.solver_config.brkga_elite_bias:
                        child_assignment.append(elite_gene)
                    else:
                        child_assignment.append(non_elite_gene)
                next_population.append(
                    Chromosome(
                        priority_genes=child_priority,
                        assignment_genes=child_assignment,
                    )
                )

            population = next_population

        if best_chromosome is None or best_eval is None:
            raise RuntimeError("BRKGA failed to produce a solution.")
        return clone_routes(best_chromosome.routes or []), best_eval

    def _build_initial_population(self, alns_routes: Routes) -> List[Chromosome]:
        population: List[Chromosome] = []
        warm_count = max(
            1,
            int(
                self.solver_config.brkga_warm_start_proportion
                * self.solver_config.brkga_population_size
            ),
        )

        seed_routes = [
            alns_routes,
            self._heuristic_nearest_neighbor(),
            self._heuristic_load_balanced(),
            self._heuristic_spt(),
        ]
        seeded = 0
        while seeded < min(self.solver_config.brkga_warm_seed_count, warm_count):
            source = seed_routes[seeded % len(seed_routes)]
            chromosome = self._encode_solution(source)
            chromosome = self._perturb_chromosome(chromosome)
            chromosome.fitness, chromosome.routes = self._decode_chromosome(chromosome)
            population.append(chromosome)
            seeded += 1

        while len(population) < self.solver_config.brkga_population_size:
            population.append(self._random_chromosome())

        return population

    def _decode_chromosome(self, chromosome: Chromosome) -> Tuple[float, Routes]:
        operation_order = sorted(
            enumerate(self.operation_gene_order),
            key=lambda item: (
                chromosome.priority_genes[item[0]],
                item[1].customer_id,
                item[1].kind,
            ),
        )
        scheduled = [False] * len(self.operation_gene_order)
        routes: Routes = [[] for _ in range(self.instance.vehicle_count)]
        times = [0.0 for _ in range(self.instance.vehicle_count)]
        loads = [self.instance.capacity for _ in range(self.instance.vehicle_count)]
        locations = [self.instance.depot_node for _ in range(self.instance.vehicle_count)]
        drop_times: Dict[int, float] = {}

        for _ in range(2 * len(self.customers)):
            progress = False
            for gene_index, operation in operation_order:
                if scheduled[gene_index]:
                    continue

                feasible_vehicles = []
                if operation.kind == "D":
                    feasible_vehicles = [
                        vehicle_id
                        for vehicle_id in range(self.instance.vehicle_count)
                        if loads[vehicle_id] > 0
                    ]
                else:
                    if operation.customer_id not in drop_times:
                        continue
                    ready_time = (
                        drop_times[operation.customer_id]
                        + self.instance.processing_times[operation.customer_id]
                    )
                    feasible_vehicles = [
                        vehicle_id
                        for vehicle_id in range(self.instance.vehicle_count)
                        if loads[vehicle_id] < self.instance.capacity
                    ]

                if not feasible_vehicles:
                    continue

                hinted_vehicle = min(
                    self.instance.vehicle_count - 1,
                    int(chromosome.assignment_genes[gene_index] * self.instance.vehicle_count),
                )
                station_node = self.customer_node(operation.customer_id)
                selected_vehicle = min(
                    feasible_vehicles,
                    key=lambda vehicle_id: (
                        self._decoder_completion_time(
                            times[vehicle_id],
                            locations[vehicle_id],
                            station_node,
                            operation.customer_id,
                            ready_time if operation.kind == "P" else None,
                        ),
                        abs(vehicle_id - hinted_vehicle),
                        vehicle_id,
                    ),
                )

                arrival = times[selected_vehicle] + self.travel_time(
                    locations[selected_vehicle], station_node
                )
                routes[selected_vehicle].append(operation)
                locations[selected_vehicle] = station_node
                if operation.kind == "D":
                    times[selected_vehicle] = arrival
                    loads[selected_vehicle] -= 1
                    drop_times[operation.customer_id] = arrival
                else:
                    times[selected_vehicle] = max(arrival, ready_time)
                    loads[selected_vehicle] += 1
                scheduled[gene_index] = True
                progress = True

            if all(scheduled):
                break
            if not progress:
                continue

        base_fitness = max(
            (
                times[vehicle_id]
                + self.travel_time(locations[vehicle_id], self.instance.depot_node)
            )
            for vehicle_id in range(self.instance.vehicle_count)
        )
        unscheduled = scheduled.count(False)
        if unscheduled:
            return (
                base_fitness
                + (self.solver_config.brkga_infeasibility_penalty * unscheduled),
                routes,
            )

        evaluated = self._evaluate_cost(routes, require_complete=True)
        if not evaluated.feasible:
            return base_fitness + self.solver_config.brkga_infeasibility_penalty, routes
        return evaluated.makespan, routes

    def _decoder_completion_time(
        self,
        current_time: float,
        current_node: str,
        target_node: str,
        customer_id: int,
        ready_time: float | None,
    ) -> float:
        arrival = current_time + self.travel_time(current_node, target_node)
        if ready_time is None:
            return arrival
        # Assumption: we follow the paper's MILP and ALNS timing semantics
        # where a pickup may arrive early and wait until T_drop[c] + p_c.
        return max(arrival, ready_time)

    def _encode_solution(self, routes: Routes) -> Chromosome:
        priority = [0.999999 for _ in self.operation_gene_order]
        assignment = [0.0 for _ in self.operation_gene_order]
        for vehicle_id, route in enumerate(routes):
            route_length = max(1, len(route))
            for rank, operation in enumerate(route, start=1):
                index = self.gene_index[(operation.customer_id, operation.kind)]
                priority[index] = (rank - 1) / route_length
                assignment[index] = vehicle_id / self.instance.vehicle_count
        return Chromosome(priority_genes=priority, assignment_genes=assignment)

    def _random_chromosome(self) -> Chromosome:
        return Chromosome(
            priority_genes=[self.rng.random() for _ in self.operation_gene_order],
            assignment_genes=[self.rng.random() for _ in self.operation_gene_order],
        )

    def _perturb_chromosome(self, chromosome: Chromosome) -> Chromosome:
        perturbed_priority = [
            clamp01(
                gene
                + self.rng.uniform(
                    -self.solver_config.brkga_gene_perturbation,
                    self.solver_config.brkga_gene_perturbation,
                )
            )
            for gene in chromosome.priority_genes
        ]
        perturbed_assignment = [
            clamp01(
                gene
                + self.rng.uniform(
                    -self.solver_config.brkga_gene_perturbation,
                    self.solver_config.brkga_gene_perturbation,
                )
            )
            for gene in chromosome.assignment_genes
        ]
        return Chromosome(
            priority_genes=perturbed_priority,
            assignment_genes=perturbed_assignment,
        )

    # ------------------------------------------------------------------
    # Warm-start construction heuristics
    # ------------------------------------------------------------------

    def _heuristic_nearest_neighbor(self) -> Routes:
        routes: Routes = [[] for _ in range(self.instance.vehicle_count)]
        loads = [self.instance.capacity for _ in range(self.instance.vehicle_count)]
        current_nodes = [self.instance.depot_node for _ in range(self.instance.vehicle_count)]
        current_times = [0.0 for _ in range(self.instance.vehicle_count)]
        assignments = defaultdict(list)

        remaining = set(self.customers)
        while remaining:
            best = None
            for customer_id in remaining:
                station_node = self.customer_node(customer_id)
                for vehicle_id in range(self.instance.vehicle_count):
                    if loads[vehicle_id] <= 0:
                        continue
                    completion = current_times[vehicle_id] + self.travel_time(
                        current_nodes[vehicle_id], station_node
                    )
                    candidate = (completion, vehicle_id, customer_id)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                break
            _, vehicle_id, customer_id = best
            assignments[vehicle_id].append(customer_id)
            current_times[vehicle_id] += self.travel_time(
                current_nodes[vehicle_id], self.customer_node(customer_id)
            )
            current_nodes[vehicle_id] = self.customer_node(customer_id)
            loads[vehicle_id] -= 1
            remaining.remove(customer_id)

        for vehicle_id, cluster in assignments.items():
            routes[vehicle_id] = self._build_cluster_route(cluster)
        return routes

    def _heuristic_load_balanced(self) -> Routes:
        ordered = sorted(self.customers, key=self.customer_workload, reverse=True)
        clusters = [[] for _ in range(self.instance.vehicle_count)]
        loads = [0.0 for _ in range(self.instance.vehicle_count)]
        for customer_id in ordered:
            vehicle_id = min(range(self.instance.vehicle_count), key=lambda idx: loads[idx])
            clusters[vehicle_id].append(customer_id)
            loads[vehicle_id] += self.customer_workload(customer_id)
        return [self._build_cluster_route(cluster) for cluster in clusters]

    def _heuristic_spt(self) -> Routes:
        ordered = sorted(self.customers, key=lambda customer_id: self.instance.processing_times[customer_id])
        clusters = [[] for _ in range(self.instance.vehicle_count)]
        for index, customer_id in enumerate(ordered):
            clusters[index % self.instance.vehicle_count].append(customer_id)
        return [self._build_cluster_route(cluster) for cluster in clusters]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def customer_node(self, customer_id: int) -> str:
        return self.instance.station_by_id[customer_id].node_id

    def customer_workload(self, customer_id: int) -> float:
        return (
            2.0 * self.travel_time(self.instance.depot_node, self.customer_node(customer_id))
            + self.instance.processing_times[customer_id]
        )

    def travel_time(self, source: str, target: str) -> float:
        return self.travel_times_dense[self.node_index[source]][self.node_index[target]]

    def path_coords(self, source: str, target: str) -> List[Tuple[float, float]]:
        node_path = self.instance.world.shortest_paths[(source, target)]
        return self._lane_polyline(node_path)

    def lane_polyline(self, node_path: Sequence[str]) -> List[Tuple[float, float]]:
        """Public lane-adjusted polyline for an arbitrary node path."""
        return self._lane_polyline(node_path)

    def _lane_polyline(self, node_path: Sequence[str]) -> List[Tuple[float, float]]:
        if not node_path:
            return []
        if len(node_path) == 1:
            return [self.instance.world.coords[node_path[0]]]

        points: List[Tuple[float, float]] = []
        for index, node_id in enumerate(node_path):
            coord = self.instance.world.coords[node_id]
            if node_id == self.instance.depot_node or node_id.startswith("station_"):
                points.append(coord)
                continue

            prev_dir = self._segment_direction(node_path[index - 1], node_id) if index > 0 else None
            next_dir = (
                self._segment_direction(node_id, node_path[index + 1])
                if index + 1 < len(node_path)
                else None
            )
            prev_dir = prev_dir or next_dir
            next_dir = next_dir or prev_dir
            points.append(self._lane_point(coord, prev_dir, next_dir))

        return dedupe_points(points)

    def _lane_point(
        self,
        coord: Tuple[float, float],
        prev_dir: str | None,
        next_dir: str | None,
    ) -> Tuple[float, float]:
        x, y = coord
        lane = self.instance.world.lane_center_offset_in
        vertical_dir = None
        horizontal_dir = None
        for direction in (prev_dir, next_dir):
            if direction in {"N", "S"}:
                vertical_dir = direction
            elif direction in {"E", "W"}:
                horizontal_dir = direction

        if vertical_dir and horizontal_dir:
            x_offset = -lane if vertical_dir == "N" else lane
            y_offset = lane if horizontal_dir == "E" else -lane
            return (x + x_offset, y + y_offset)
        if vertical_dir:
            x_offset = -lane if vertical_dir == "N" else lane
            return (x + x_offset, y)
        if horizontal_dir:
            y_offset = lane if horizontal_dir == "E" else -lane
            return (x, y + y_offset)
        return coord

    def _segment_direction(self, source: str, target: str) -> str | None:
        x1, y1 = self.instance.world.coords[source]
        x2, y2 = self.instance.world.coords[target]
        if math.isclose(x1, x2) and math.isclose(y1, y2):
            return None
        if math.isclose(x1, x2):
            return "N" if y2 > y1 else "S"
        if math.isclose(y1, y2):
            return "E" if x2 > x1 else "W"
        return None

    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        names = list(weights.keys())
        values = [weights[name] for name in names]
        return self.rng.choices(names, weights=values, k=1)[0]


def clone_routes(routes: Routes | None) -> Routes:
    if not routes:
        return []
    return [[Operation(op.customer_id, op.kind) for op in route] for route in routes]


def remove_customer(routes: Routes, customer_id: int) -> Routes:
    return [
        [op for op in route if op.customer_id != customer_id]
        for route in clone_routes(routes)
    ]


def remove_pickup(routes: Routes, customer_id: int) -> Routes:
    return [
        [op for op in route if not (op.customer_id == customer_id and op.kind == "P")]
        for route in clone_routes(routes)
    ]


def vehicle_customers(route: Sequence[Operation]) -> set[int]:
    return {op.customer_id for op in route}


def customers_with_operation_on_vehicle(routes: Routes, vehicle_id: int) -> set[int]:
    return {op.customer_id for op in routes[vehicle_id]}


def clamp01(value: float) -> float:
    return min(0.999999, max(0.0, value))


def dedupe_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    deduped: List[Tuple[float, float]] = []
    for point in points:
        if not deduped or not (
            math.isclose(point[0], deduped[-1][0]) and math.isclose(point[1], deduped[-1][1])
        ):
            deduped.append(point)
    return deduped
