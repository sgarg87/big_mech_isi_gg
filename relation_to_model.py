import constants_darpa_json_format as cd
import model_relation_types as mrt
import darpa_interaction_types as dit
from config_console_output import *


def compare_entity_text(entity_text1, entity_text2):
    if entity_text1.lower() == entity_text2.lower():
        return True
    if entity_text1.lower() in entity_text2.lower():
        return True
    if entity_text2.lower() in entity_text1.lower():
        return True
    return False


def compare_entity_identifiers(entity_identifier1, entity_identifier2):
    if entity_identifier1 is None or not entity_identifier1.strip():
        return False
    if entity_identifier2 is None or not entity_identifier2.strip():
        return False
    if entity_identifier1.lower() == entity_identifier2.lower():
        return True
    if entity_identifier1.lower() in entity_identifier2.lower():
        return True
    if entity_identifier2.lower() in entity_identifier1.lower():
        return True
    return False


def compare_participants(participant_1, participant_2):
    if compare_entity_text(participant_1[cd.entity_text], participant_2[cd.entity_text]):
        return True
    if compare_entity_identifiers(participant_1[cd.identifier], participant_2[cd.identifier]):
        return True
    return False


def compare_pairs(pair1_1, pair1_2, pair2_1, pair2_2):
    if compare_participants(pair1_1, pair2_1)\
            and compare_participants(pair1_2, pair2_2):
        return True
    elif compare_participants(pair1_2, pair2_1)\
            and compare_participants(pair1_1, pair2_2):
        return True
    return False


def compare_json_obj_to_model_json_objs(curr_json_obj, model_json_objs):
    # by default, current json object is extension
    curr_json_obj[cd.model_relation] = mrt.Extension
    curr_json_obj[cd.model_elements] = []
    if curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.binds:
        compare_bind_json_obj_model_json_objs(curr_json_obj, model_json_objs)
    elif curr_json_obj[cd.extracted_information][cd.interaction_type] in [dit.increases, dit.decreases]:
        compare_increase_type_json_obj_model_json_objs(curr_json_obj, model_json_objs)
        if curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases:
            if curr_json_obj[cd.model_relation] == mrt.Conflicting:
                curr_json_obj[cd.model_relation] = mrt.Corroboration
            elif curr_json_obj[cd.model_relation] == mrt.Corroboration:
                curr_json_obj[cd.model_relation] = mrt.Conflicting
    elif curr_json_obj[cd.extracted_information][cd.interaction_type] in [dit.increases_activity, dit.decreases_activity]:
        compare_increase_activity_type_json_obj_model_json_objs(curr_json_obj, model_json_objs)
        if curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases_activity:
            if curr_json_obj[cd.model_relation] == mrt.Conflicting:
                curr_json_obj[cd.model_relation] = mrt.Corroboration
            elif curr_json_obj[cd.model_relation] == mrt.Corroboration:
                curr_json_obj[cd.model_relation] = mrt.Conflicting
    elif curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.translocates:
        compare_translocate_json_obj_model_json_objs(curr_json_obj, model_json_objs)
    elif curr_json_obj[cd.extracted_information][cd.interaction_type] in [dit.adds_modification]:
        compare_adds_modification_json_obj_model_json_objs(curr_json_obj, model_json_objs)
    elif curr_json_obj[cd.extracted_information][cd.interaction_type] in [dit.removes_modification]:
        pass
    else:
        NotImplementedError


def compare_bind_json_obj_model_json_objs(curr_json_obj, model_json_objs):
    if curr_json_obj[cd.extracted_information][cd.interaction_type] != dit.binds:
        raise AssertionError
    for curr_model_json_obj in model_json_objs:
        if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.binds:
            if compare_pairs(curr_json_obj[cd.extracted_information][cd.participant_a], curr_json_obj[cd.extracted_information][cd.participant_b], curr_model_json_obj[cd.extracted_information][cd.participant_a],
                             curr_model_json_obj[cd.extracted_information][cd.participant_b]):
                curr_json_obj[cd.model_relation] = mrt.Corroboration
                curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                return
        elif curr_model_json_obj[cd.extracted_information][cd.interaction_type] in [dit.increases, dit.decreases]\
                and cd.entities in curr_model_json_obj[cd.extracted_information][cd.participant_b]:
            if compare_pairs(curr_json_obj[cd.extracted_information][cd.participant_a], curr_json_obj[cd.extracted_information][cd.participant_b],
                             curr_model_json_obj[cd.extracted_information][cd.participant_b][cd.entities][0],
                             curr_model_json_obj[cd.extracted_information][cd.participant_b][cd.entities][1]):
                if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases:
                    curr_json_obj[cd.model_relation] = mrt.Conflicting
                elif curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases:
                    curr_json_obj[cd.model_relation] = mrt.Corroboration
                curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                return


def compare_increase_type_json_obj_model_json_objs(curr_json_obj, model_json_objs):
    for curr_model_json_obj in model_json_objs:
        if cd.entities in curr_json_obj[cd.extracted_information][cd.participant_b] and curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.binds:
            if compare_pairs(curr_json_obj[cd.extracted_information][cd.participant_b][cd.entities][0],
                             curr_json_obj[cd.extracted_information][cd.participant_b][cd.entities][1],
                             curr_model_json_obj[cd.extracted_information][cd.participant_a],
                             curr_model_json_obj[cd.extracted_information][cd.participant_b]):
                if curr_json_obj[cd.extracted_information][cd.participant_a] is None:
                    curr_json_obj[cd.model_relation] = mrt.Corroboration
                else:
                    curr_json_obj[cd.model_relation] = mrt.Specialization
                curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                return
        elif curr_model_json_obj[cd.extracted_information][cd.interaction_type] in [dit.increases, dit.decreases]:
            if cd.entities in curr_json_obj[cd.extracted_information][cd.participant_b] \
                    and cd.entities in curr_model_json_obj[cd.extracted_information][cd.participant_b]:
                assert curr_json_obj[cd.extracted_information][cd.participant_a] is not None
                assert curr_model_json_obj[cd.extracted_information][cd.participant_a] is not None
                if compare_pairs(curr_json_obj[cd.extracted_information][cd.participant_b][cd.entities][0],
                                 curr_json_obj[cd.extracted_information][cd.participant_b][cd.entities][1],
                                 curr_model_json_obj[cd.extracted_information][cd.participant_b][cd.entities][0],
                                 curr_model_json_obj[cd.extracted_information][cd.participant_b][cd.entities][1]):
                    if compare_participants(curr_json_obj[cd.extracted_information][cd.participant_a],
                                            curr_model_json_obj[cd.extracted_information][cd.participant_a]):
                        if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases:
                            curr_json_obj[cd.model_relation] = mrt.Corroboration
                        elif curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases:
                            curr_json_obj[cd.model_relation] = mrt.Conflicting
                        else:
                            raise AssertionError
                        curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                        return
            elif cd.entities not in curr_json_obj[cd.extracted_information][cd.participant_b]\
                    and cd.entities not in curr_model_json_obj[cd.extracted_information][cd.participant_b]:
                if compare_participants(curr_json_obj[cd.extracted_information][cd.participant_b],
                                        curr_model_json_obj[cd.extracted_information][cd.participant_b]):
                    if curr_json_obj[cd.extracted_information][cd.participant_a] is None:
                        if curr_model_json_obj[cd.extracted_information][cd.participant_a] is None:
                            if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases:
                                curr_json_obj[cd.model_relation] = mrt.Corroboration
                            elif curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases:
                                curr_json_obj[cd.model_relation] = mrt.Conflicting
                            else:
                                raise AssertionError
                            curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                            return
                        else:
                            if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases:
                                curr_json_obj[cd.model_relation] = mrt.Corroboration
                                curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                                return
                    else:
                        if curr_model_json_obj[cd.extracted_information][cd.participant_a] is None:
                            curr_json_obj[cd.model_relation] = mrt.Specialization
                            curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                            return
                        else:
                            if compare_participants(curr_json_obj[cd.extracted_information][cd.participant_a],
                                                    curr_model_json_obj[cd.extracted_information][cd.participant_a]):
                                if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases:
                                    curr_json_obj[cd.model_relation] = mrt.Corroboration
                                elif curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases:
                                    curr_json_obj[cd.model_relation] = mrt.Conflicting
                                else:
                                    raise AssertionError
                                curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                                return


def compare_increase_activity_type_json_obj_model_json_objs(curr_json_obj, model_json_objs):
    for curr_model_json_obj in model_json_objs:
        if curr_model_json_obj[cd.extracted_information][cd.interaction_type] in [dit.increases_activity, dit.decreases_activity]:
            if compare_participants(curr_json_obj[cd.extracted_information][cd.participant_b],
                                    curr_model_json_obj[cd.extracted_information][cd.participant_b]):
                if curr_json_obj[cd.extracted_information][cd.participant_a] is None:
                    if curr_model_json_obj[cd.extracted_information][cd.participant_a] is None:
                        if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases_activity:
                            curr_json_obj[cd.model_relation] = mrt.Corroboration
                        elif curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases_activity:
                            curr_json_obj[cd.model_relation] = mrt.Conflicting
                        else:
                            raise AssertionError
                        curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                        return
                    else:
                        if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases_activity:
                            curr_json_obj[cd.model_relation] = mrt.Corroboration
                            curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                            return
                else:
                    if curr_model_json_obj[cd.extracted_information][cd.participant_a] is None:
                        curr_json_obj[cd.model_relation] = mrt.Specialization
                        curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                        return
                    else:
                        if compare_participants(curr_json_obj[cd.extracted_information][cd.participant_a],
                                                curr_model_json_obj[cd.extracted_information][cd.participant_a]):
                            if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases_activity:
                                curr_json_obj[cd.model_relation] = mrt.Corroboration
                            elif curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases_activity:
                                curr_json_obj[cd.model_relation] = mrt.Conflicting
                            else:
                                raise AssertionError
                            curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                            return


def compare_translocate_json_obj_model_json_objs(curr_json_obj, model_json_objs):
    for curr_model_json_obj in model_json_objs:
        if curr_model_json_obj[cd.extracted_information][cd.interaction_type] in [dit.translocates]:
            if compare_participants(curr_json_obj[cd.extracted_information][cd.participant_b],
                                    curr_model_json_obj[cd.extracted_information][cd.participant_b]):
                if curr_json_obj[cd.extracted_information][cd.participant_a] is None:
                    curr_json_obj[cd.model_relation] = mrt.Corroboration
                    curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                    return
                else:
                    if curr_model_json_obj[cd.extracted_information][cd.participant_a] is None:
                        curr_json_obj[cd.model_relation] = mrt.Specialization
                        curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                        return
                    else:
                        if compare_participants(curr_json_obj[cd.extracted_information][cd.participant_a],
                                                curr_model_json_obj[cd.extracted_information][cd.participant_a]):
                            curr_json_obj[cd.model_relation] = mrt.Corroboration
                            curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                            return


def compare_adds_modification_json_obj_model_json_objs(curr_json_obj, model_json_objs):
    # todo: considering only a single modification currently
    if curr_json_obj[cd.extracted_information][cd.interaction_type] != dit.adds_modification:
        raise AssertionError
    if len(curr_json_obj[cd.extracted_information][cd.modification]) > 1:
        raise NotImplementedError
    curr_json_modification_type = curr_json_obj[cd.extracted_information][cd.modification][0][cd.modification_type]
    curr_json_modification_position = curr_json_obj[cd.extracted_information][cd.modification][0][cd.position]
    for curr_model_json_obj in model_json_objs:
        if curr_model_json_obj[cd.extracted_information][cd.interaction_type] not in [dit.adds_modification, dit.removes_modification]:
            continue
        relation = None
        if compare_participants(curr_json_obj[cd.extracted_information][cd.participant_b],
                                curr_model_json_obj[cd.extracted_information][cd.participant_b]):
            if curr_json_obj[cd.extracted_information][cd.participant_a] is None:
                relation = mrt.Corroboration
            else:
                if curr_model_json_obj[cd.extracted_information][cd.participant_a] is None:
                    relation = mrt.Specialization
                else:
                    if compare_participants(curr_json_obj[cd.extracted_information][cd.participant_a],
                                            curr_model_json_obj[cd.extracted_information][cd.participant_a]):
                        relation = mrt.Corroboration
        if relation is not None:
            if relation == mrt.Corroboration:
                if curr_model_json_obj[cd.extracted_information][cd.interaction_type] == dit.removes_modification:
                    relation = mrt.Conflicting
            is_modification_match = False
            for curr_model_modification in curr_model_json_obj[cd.extracted_information][cd.modification]:
                if curr_model_modification[cd.modification_type] == curr_json_modification_type:
                    is_modification_match = True
                    break
            if is_modification_match:
                curr_json_obj[cd.model_relation] = relation
                curr_json_obj[cd.model_elements] = [curr_model_json_obj[cd.biopax_id]]
                break
