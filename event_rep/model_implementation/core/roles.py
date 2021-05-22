"""
File: event_rep/model_implementation/core/roles.py
Author: Mughilan Muthupari
Creation Date: 2021-05-15

This file contains all of our role definitions.
"""


class Roles:
    has_role_other = True
    has_mod_other = False

    ROLE_PRD = "PRD"
    TEST_ROLE_PRD = "V"

    ROLE_OTHER = "ARG-OTHER" if has_role_other else None  # make it None if not using ROLE_OTHER
    TEST_ROLE_OTHER = '<OTHER>' if has_role_other else None  # "A-REST"

    TEST_ROLES = {
        'A0': 5,
        'A1': 1,
        'A2': 4,
        'A3': 10,
        'A4': 14,
        'A5': 28,
        'AM': 25,
        'AM-ADV': 12,
        'AM-CAU': 8,
        'AM-DIR': 18,
        'AM-DIS': 6,
        'AM-EXT': 24,
        'AM-LOC': 2,
        'AM-MNR': 7,
        'AM-MOD': 21,
        'AM-NEG': 13,
        'AM-PNC': 15,
        'AM-PRD': 27,
        'AM-TMP': 3,
        'C-A1': 26,
        'C-V': 9,
        'R-A0': 11,
        'R-A1': 17,
        'R-A2': 20,
        'R-A3': 29,
        'R-AM-CAU': 23,
        'R-AM-LOC': 16,
        'R-AM-MNR': 19,
        'R-AM-TMP': 22,
        '<OTHER>': 30
    }

    @staticmethod
    def isModifier(role):
        return ((len(role) >= 2) and (role[:2] == "AM")) or ((len(role) >= 4) and (role[:4] == "ARGM"))

    @staticmethod
    def adjustRole(role):
        return role


class Roles4Args3Mods2Others(Roles):
    has_role_other = True
    has_mod_other = True
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, "ARG3-4-5": 7, "ARGM-OTHER": 8, Roles.ROLE_OTHER: 9}
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, "A3-4-5": 7, "AM-OTHER": 8, Roles.TEST_ROLE_OTHER: 9}

    @staticmethod
    def adjustRole(role):
        if role in ("ARG3", "ARG4", "ARG5"):
            role = "ARG3-4-5"
        elif (role[:4] == "ARGM") and (role not in ("ARGM-LOC", "ARGM-TMP", "ARGM-MNR")):
            role = "ARGM-OTHER"
        elif (role[:2] == "AM") and (role not in ("AM-LOC", "AM-TMP", "AM-MNR")):
            role = "AM-OTHER"
        return role


class Roles2Args3Mods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5, }
    #         "ARG2": 6, "ARG3-4-5":7, "ARGM-OTHER": 8} # ROLE_OTHER: 9}
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5, }
    #              "A2": 6, "A3-4-5":7, "AM-OTHER": 8, "<UNKNOWN>": 9}


class Roles2Args3Mods1Other(Roles):
    has_role_other = True
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   Roles.ROLE_OTHER: 6}
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  Roles.TEST_ROLE_OTHER: 6}


class Roles3Args3Mods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, }
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, }


class Roles3Args4Mods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, "ARGM-MOD": 7, }
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, "AM-MOD": 7, }


class Roles3Args5Mods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, "ARGM-MOD": 7, "ARGM-ADV": 8, }
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, "AM-MOD": 7, "AM-ADV": 8, }


class Roles3Args6Mods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, "ARGM-MOD": 7, "ARGM-ADV": 8, "ARGM-DIS": 9, }
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, "AM-MOD": 7, "AM-ADV": 8, "AM-DIS": 9, }


class Roles3Args7Mods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, "ARGM-MOD": 7, "ARGM-ADV": 8, "ARGM-DIS": 9, "ARGM-NEG": 10, }
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, "AM-MOD": 7, "AM-ADV": 8, "AM-DIS": 9, "AM-NEG": 10, }


class Roles4Args3Mods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, "ARG3": 7, }
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, "A3": 7, }


class Roles5Args3Mods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, "ARG3": 7, "ARG4": 8, }
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, "A3": 7, "A4": 8, }


class Roles6Args3Mods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, "ARG3": 7, "ARG4": 8, "ARG5": 9, }
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, "A3": 7, "A4": 8, "A5": 9, }


class RolesAllLsgnArgsMods(Roles):
    has_role_other = False
    has_mod_other = False
    TRAIN_ROLES = {"ARG0": 0, "ARG1": 1, "ARGM-LOC": 2, "ARGM-TMP": 3, "ARGM-MNR": 4, Roles.ROLE_PRD: 5,
                   "ARG2": 6, "ARG3": 7, "ARG4": 8, "ARG5": 9,
                   "ARGM-MOD": 10, "ARGM-ADV": 11, "ARGM-DIS": 12, "ARGM-NEG": 13,
                   "ARGM-PRP": 14, "ARGM-DIR": 15, "ARGM-ADJ": 16, "ARGM-CAU": 17, "ARGM-EXT": 18,
                   "ARGM-PRD": 19, "ARGM-LVB": 20, "ARGM-GOL": 21, "ARGM-COM": 22, "ARGM-PNC": 23, "ARGM-REC": 24, }
    TEST_ROLES = {"A0": 0, "A1": 1, "AM-LOC": 2, "AM-TMP": 3, "AM-MNR": 4, Roles.TEST_ROLE_PRD: 5,
                  "A2": 6, "A3": 7, "A4": 8, "A5": 9,
                  "AM-MOD": 10, "AM-ADV": 11, "AM-DIS": 12, "AM-NEG": 13,
                  "AM-PRP": 14, "AM-DIR": 15, "AM-ADJ": 16, "AM-CAU": 17, "AM-EXT": 18,
                  "AM-PRD": 19, "AM-LVB": 20, "AM-GOL": 21, "AM-COM": 22, "AM-PNC": 23, "AM-REC": 24, }


class Anims:
    ANIM_NONE = "None"
    ANIMS = {ANIM_NONE: 0, "Animate": 1, "Sentient": 2}


class Senti2:
    ANIM_NONE = u"None"
    ANIM_PRD = u"ANIM_PRD"
    ANIMS = {ANIM_NONE: 0, ANIM_PRD: 3,
             u"Animate?": 4, u"Animate": 1, u"Animates?": 5, u"Animates": 6,
             u"Sentient?": 7, u"Sentient": 2, u"Sentients?": 8, u"Sentients": 9,
             u"InanimPron": 10, u"SentiPron?": 11, u"SentiPron": 12, }
    # "InanimPron" should perhaps better be named "AnimPron?" because can refer to animals or inaminate


class Senti2_3vals:
    ANIM_NONE = "None"
    ANIM_PRD = "ANIM_PRD"
    ANIMS = {ANIM_NONE: 0, ANIM_PRD: 3,
             "Animate?": 1, "Animate": 1, "Animates?": 1, "Animates": 1,
             "Sentient?": 2, "Sentient": 2, "Sentients?": 2, "Sentients": 2,
             "InanimPron": 0, "SentiPron?": 2, "SentiPron": 2, }
    # "InanimPron" should perhaps better be named "AnimPron?" because can refer to animals or inaminate


# coding "InanimPron" ("it") as animate (1)
class Senti2_3valsB:
    ANIM_NONE = "None"
    ANIM_PRD = "ANIM_PRD"
    ANIMS = {ANIM_NONE: 0, ANIM_PRD: 3,
             "Animate?": 1, "Animate": 1, "Animates?": 1, "Animates": 1,
             "Sentient?": 2, "Sentient": 2, "Sentients?": 2, "Sentients": 2,
             "InanimPron": 1, "SentiPron?": 2, "SentiPron": 2, }
    # "InanimPron" should perhaps better be named "AnimPron?" because can refer to animals or inaminate


# groups (Animates, Sentients) get separate code from individuals; "InanimPron" ("it") get a separate code
class Senti2_6vals_grp:
    ANIM_NONE = "None"
    ANIM_PRD = "ANIM_PRD"
    ANIMS = {ANIM_NONE: 0, ANIM_PRD: 3,
             "Animate?": 1, "Animate": 1, "Animates?": 4, "Animates": 4,
             "Sentient?": 2, "Sentient": 2, "Sentients?": 5, "Sentients": 5,
             "InanimPron": 6, "SentiPron?": 2, "SentiPron": 2, }
    # "InanimPron" should perhaps better be named "AnimPron?" because can refer to animals or inaminate


# noisy q marks ("?") values get separate code, but groups together with indiv
class Senti2_6vals_qm:
    ANIM_NONE = "None"
    ANIM_PRD = "ANIM_PRD"
    ANIMS = {ANIM_NONE: 0, ANIM_PRD: 3,
             "Animate?": 4, "Animate": 1, "Animates?": 4, "Animates": 1,
             "Sentient?": 5, "Sentient": 2, "Sentients?": 5, "Sentients": 2,
             "InanimPron": 6, "SentiPron?": 5, "SentiPron": 2, }
    # "InanimPron" should perhaps better be named "AnimPron?" because can refer to animals or inaminate


# merge ambig pronouns, and ignore other "?" cases
class Senti2_4vals:
    ANIM_NONE = "None"
    ANIM_PRD = "ANIM_PRD"
    ANIMS = {ANIM_NONE: 0, ANIM_PRD: 3,
             "Animate?": 0, "Animate": 1, "Animates?": 0, "Animates": 1,
             "Sentient?": 0, "Sentient": 2, "Sentients?": 0, "Sentients": 2,
             "InanimPron": 4, "SentiPron?": 4, "SentiPron": 2, }
    # "InanimPron" should perhaps better be named "AnimPron?" because can refer to animals or inaminate