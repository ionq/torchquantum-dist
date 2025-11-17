from typing import Union

import torch
import torch.distributed
from torch.distributed.tensor import DTensor


def interchange_qubits(
    state: Union[DTensor, torch.Tensor], grouping: torch.Tensor, wire1: int, wire2: int
) -> tuple[Union[DTensor, torch.Tensor], torch.Tensor]:
    """
    Interchanges two qubits within a given statevector. Does not reshard on its own and can only interchange sharded qubits with ungrouped qubits.
    Operates recurseively based on a few base interchange patterns.
    Args:
        state: Tensor or DTensor containing state information
        grouping: grouping information for state
        wire1: first qubit index
        wire2: second qubit index
    Returns:
        new_state: Tensor or DTensor with interchanged qubits
        new_grouping: new grouping information for new_state
    """
    wire_info = grouping[:, [wire1, wire2]]
    if wire1 == wire2:
        return state, grouping
    elif (wire_info[1] > -1).any():
        if (
            wire_info[1] == -2
        ).any():  # Cannot interchange sharded qubit with grouped unsharded qubit
            return state, grouping
        elif (wire_info[1] == -1).any():  # Exactly one wire is ungrouped
            if (
                wire_info[1, 0] == -1
            ):  # Presume from now on that wire2 is the ungrouped wire
                return interchange_qubits(state, grouping, wire2, wire1)
            else:
                lone_wire_idx = wire_info[0, 1]
                # Get index and size of wire1 group along with relative position of wire1
                grouped_wire_info = [
                    wire_info[0, 0].item(),
                    wire_info[1, 0].item(),
                    (grouping[0] == wire_info[0, 0]).sum().item(),
                ]
                # grouped wires cannot be leftmost or rightmost dimensions; get index and size of left and right groups
                left_group_info = [
                    grouped_wire_info[0] - 1,
                    (grouping[0] == (grouped_wire_info[0] - 1)).sum().item(),
                ]
                right_group_info = [
                    grouped_wire_info[0] + 1,
                    (grouping[0] == (grouped_wire_info[0] + 1)).sum().item(),
                ]

                need_left_interchange = (lone_wire_idx == left_group_info[0]) and (
                    grouped_wire_info[1] > 0
                )
                need_right_interchange = (lone_wire_idx == right_group_info[0]) and (
                    grouped_wire_info[1] < grouped_wire_info[2] - 1
                )
                need_interchange = need_left_interchange or need_right_interchange
                if need_interchange:
                    all_ungrouped_idxs = grouping[0, grouping[1] == -1]
                    # Pick a new ungrouped qubit that isn't adjacent to wire1
                    helper_idx = all_ungrouped_idxs[
                        (
                            (all_ungrouped_idxs != left_group_info[0])
                            & (all_ungrouped_idxs != right_group_info[0])
                        )
                    ][0]
                    helper_qubit = (
                        torch.nonzero(grouping[0] == helper_idx).flatten()[0]
                    ).int()
                    new_state, new_grouping = interchange_qubits(
                        state, grouping, wire1, helper_qubit
                    )
                    new_state, new_grouping = interchange_qubits(
                        new_state, new_grouping, wire1, wire2
                    )
                    new_state, new_grouping = interchange_qubits(
                        new_state, new_grouping, helper_qubit, wire2
                    )
                    return new_state, new_grouping
                else:
                    # Get temporary tensor shape
                    state_shape = list(state.shape)
                    temp_shape = state_shape.copy()
                    temp_shape[grouped_wire_info[0]] = 2
                    temp_shape[left_group_info[0]] = 2 ** (
                        left_group_info[1] + grouped_wire_info[1]
                    )
                    temp_shape[right_group_info[0]] = 2 ** (
                        right_group_info[1]
                        + grouped_wire_info[2]
                        - 1
                        - grouped_wire_info[1]
                    )

                    # Reshape to isolate grouped wire dimension, interchange wires, and then reshape back
                    new_state = state.reshape(temp_shape)
                    new_state, _ = interchange_dims(
                        new_state,
                        grouping,
                        grouped_wire_info[0],
                        lone_wire_idx,
                        new_state.ndim,
                    )
                    new_state = new_state.reshape(state_shape)

                    new_grouping = grouping.detach().clone()
                    new_grouping[:, wire1], new_grouping[:, wire2] = (
                        grouping[:, wire2],
                        grouping[:, wire1],
                    )
                    return new_state, new_grouping
        else:  # If both wires are grouped, use ungrouped wire as medium of interchange
            helper_qubit = (torch.nonzero(grouping[1] == -1).flatten()[0]).int()
            new_state, new_grouping = interchange_qubits(
                state, grouping, wire2, helper_qubit
            )
            new_state, new_grouping = interchange_qubits(
                new_state, new_grouping, wire1, wire2
            )
            new_state, new_grouping = interchange_qubits(
                new_state, new_grouping, helper_qubit, wire1
            )
            return new_state, new_grouping
    else:
        # Between ungrouped and sharded qubits, interchanging qubits is the same as interchanging dimensions
        return interchange_dims(
            state, grouping, wire_info[0, 0], wire_info[0, 1], state.ndim
        )


def interchange_dims(
    state: Union[DTensor, torch.Tensor],
    grouping: torch.Tensor,
    dim1: int,
    dim2: int,
    num_dims: int,
) -> tuple[Union[DTensor, torch.Tensor], torch.Tensor]:
    """
    Interchanges two tensor dimensions, either groups of qubits or ungrouped qubits, and updates grouping tensor
    Note that relative orders within dimensions (or designations of ungrouped or sharded qubits, remain unchanged
    Args:
        state: Tensor or DTensor containing state info. Must contain num_dims dims
        grouping: Tensor containing grouping info for state
        dim1: first dimension
        dim2: second dimension
        num_dims: number of dimensions in states
    Results:
        new_state: new state with dimensions interchanged
        new_grouping: new grouping with dimensions interchanged
    """
    permute_list = list(range(num_dims))
    permute_list[dim1], permute_list[dim2] = permute_list[dim2], permute_list[dim1]
    new_state = state.permute(permute_list)
    new_grouping = grouping.detach().clone()
    where_dim1 = new_grouping[0] == dim1
    where_dim2 = new_grouping[0] == dim2
    new_grouping[0, where_dim1], new_grouping[0, where_dim2] = dim2, dim1
    return new_state, new_grouping
