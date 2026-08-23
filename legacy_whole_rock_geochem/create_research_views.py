#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建研究子数据集视图
为不同研究目的创建预定义的数据视图，不修改原始数据

年龄有效性规则：
  - age IS NULL（未知年龄）→ 保留
  - 0 < age <= 10000 → 保留
  - age <= 0 或 age > 10000 → 排除

重复性检验策略：
  - 多维度独立打分：空间 + 主量 + 微量 + 同位素
  - 缺失维度跳过，不惩罚得分
  - 同文献内重复保留（属采样设计）
  - 跨文献重复组内，保留字段完整度最高的样本
"""

import sqlite3
from pathlib import Path
from collections import defaultdict
import hashlib
import json

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"
DB_PATH = DATA_DIR / "geochem.db"

# ─────────────────────────────────────────────────────────────────────────────
# 年龄条件（统一常量，所有视图使用同一规则）
# 规则：NULL 视为有效；有值时必须 > 0 且 <= 10000
# ─────────────────────────────────────────────────────────────────────────────
AGE_CONDITION = "(a.age IS NULL OR (a.age > 0 AND a.age <= 10000))"


def connect() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH))


# =============================================================================
# 属性层 canonical mapping
# =============================================================================

def _row_hash(values: tuple) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_canonical_map(
    conn: sqlite3.Connection,
    table_name: str,
    id_col: str,
    compare_cols: list[str],
    map_table_name: str,
    canonical_col: str,
) -> None:
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {map_table_name}")
    cursor.execute(
        f"""
        CREATE TABLE {map_table_name} (
            {id_col} INTEGER PRIMARY KEY,
            {canonical_col} INTEGER NOT NULL,
            duplicate_group_hash TEXT
        )
        """
    )

    rows = cursor.execute(
        f"SELECT {id_col}, {', '.join(compare_cols)} FROM {table_name}"
    ).fetchall()

    groups = defaultdict(list)
    for row in rows:
        rid = row[0]
        values = row[1:]
        groups[_row_hash(values)].append(rid)

    inserts = []
    duplicate_groups = 0
    duplicate_rows = 0
    for group_hash, ids in groups.items():
        canonical_id = min(ids)
        if len(ids) > 1:
            duplicate_groups += 1
            duplicate_rows += len(ids)
        for rid in ids:
            inserts.append((rid, canonical_id, group_hash if len(ids) > 1 else None))

    cursor.executemany(
        f"INSERT INTO {map_table_name} ({id_col}, {canonical_col}, duplicate_group_hash) VALUES (?,?,?)",
        inserts,
    )
    conn.commit()
    print(
        f"  ✓ {map_table_name}: {len(inserts)} 行, 冗余组 {duplicate_groups} 个, 冗余行 {duplicate_rows} 条"
    )


def build_canonical_id_maps(conn: sqlite3.Connection) -> None:
    print("\n[辅助] 构建属性层 canonical mapping ...")
    cursor = conn.cursor()

    specs = [
        ("reference", "ref_id", ["author", "title", "journal", "year", "doi", "bibtex", "data_source"],
         "canonical_reference_map", "canonical_ref_id"),
        ("method", "method_id", ["method"], "canonical_method_map", "canonical_method_id"),
        ("age", "age_id", ["age_min", "age", "age_max", "age_sd", "time_period_min", "time_period", "time_period_max", "age_method"],
         "canonical_age_map", "canonical_age_id"),
    ]

    for table_name, id_col, compare_cols, map_table_name, canonical_col in specs:
        _build_canonical_map(conn, table_name, id_col, compare_cols, map_table_name, canonical_col)

    cursor.execute("DROP VIEW IF EXISTS sample_canonical_ids")
    cursor.execute(
        """
        CREATE VIEW sample_canonical_ids AS
        SELECT
            s.*,
            COALESCE(crm.canonical_ref_id, s.ref_id) AS canonical_ref_id,
            COALESCE(cmm.canonical_method_id, s.method_id) AS canonical_method_id,
            COALESCE(cam.canonical_age_id, s.age_id) AS canonical_age_id
        FROM sample s
        LEFT JOIN canonical_reference_map crm ON s.ref_id = crm.ref_id
        LEFT JOIN canonical_method_map cmm ON s.method_id = cmm.method_id
        LEFT JOIN canonical_age_map cam ON s.age_id = cam.age_id
        """
    )
    conn.commit()
    print("  ✓ sample_canonical_ids: 已创建")


# =============================================================================
# 去重决策表构建
# =============================================================================

def _same_dim(r1: dict, r2: dict, cols: list, min_compared: int = 1) -> bool:
    """
    判断两个样本在指定列上是否匹配。
    NULL 视为通配（跳过），实际比较的列数须达到 min_compared 才有效。
    所有被比较的列必须全部相等才返回 True。
    """
    score = 0
    compared = 0
    for col in cols:
        v1, v2 = r1.get(col), r2.get(col)
        # pandas NaN / None 都视为缺失
        if v1 is None or v2 is None:
            continue
        try:
            import math
            if math.isnan(float(v1)) or math.isnan(float(v2)):
                continue
        except (TypeError, ValueError):
            pass
        compared += 1
        if v1 == v2:
            score += 1
    return compared >= min_compared and score == compared


def _find(parent: list, x: int) -> int:
    """路径压缩 Union-Find。"""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list, x: int, y: int) -> None:
    parent[_find(parent, x)] = _find(parent, y)


def build_dedup_table(conn: sqlite3.Connection) -> None:
    """
    构建去重决策表 dedup_decisions。

    算法：
      1. 以 ROUND(lat,2) × ROUND(lon,2) 为空间格，
         同一空间格内的样本才进入成对比较。
      2. 对每一对跨文献样本，在主量、微量、同位素三个维度分别判断是否匹配，
         缺失维度跳过（不参与投票）。
         空间格已匹配 + 任意 ≥1 个额外维度匹配 → 判定为重复。
      3. 使用 Union-Find 将重复样本合并为组。
      4. 每组内按字段完整度降序保留第一个，其余标记 is_kept=0。
      5. 未落入任何重复组的样本，is_kept=1，dup_group_id=NULL。
    """
    cursor = conn.cursor()

    print("\n[辅助] 构建去重决策表 dedup_decisions ...")

    cursor.execute("DROP TABLE IF EXISTS dedup_decisions")
    cursor.execute("""
        CREATE TABLE dedup_decisions (
            sample_id    INTEGER PRIMARY KEY,
            is_kept      INTEGER NOT NULL DEFAULT 1,
            dup_group_id TEXT,
            completeness INTEGER DEFAULT 0
        )
    """)

    # ------------------------------------------------------------------
    # 拉取所有有坐标的样本及其模糊指纹
    # ------------------------------------------------------------------
    cursor.execute("""
        SELECT
            s.sample_id,
            s.ref_id,
            s.canonical_ref_id,
            ROUND(s.latitude,  2)  AS lat2,
            ROUND(s.longitude, 2)  AS lon2,
            -- 主量指纹（各自四舍五入降精度）
            ROUND(m.sio2,  1)  AS sio2r,
            ROUND(m.tio2,  2)  AS tio2r,
            ROUND(m.al2o3, 1)  AS al2o3r,
            ROUND(m.mgo,   1)  AS mgor,
            ROUND(m.cao,   1)  AS caor,
            ROUND(m.na2o,  2)  AS na2or,
            ROUND(m.k2o,   2)  AS k2or,
            -- 微量指纹
            ROUND(t.la_ppm, 0) AS lar,
            ROUND(t.nb_ppm, 0) AS nbr,
            ROUND(t.zr_ppm, 0) AS zrr,
            ROUND(t.sr_ppm, 0) AS srr_ppm,
            ROUND(t.th_ppm, 0) AS thr,
            -- 同位素指纹
            ROUND(i.sr87_sr86,  4) AS sr_iso,
            ROUND(i.epsilon_nd, 1) AS end_iso,
            ROUND(i.pb206_pb204,2) AS pb6_iso,
            -- 完整度计分（字段越多越完整，优先保留）
            (
              CASE WHEN m.sio2       IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN m.tio2       IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN m.al2o3      IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN m.feo        IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN m.mgo        IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN m.cao        IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN m.na2o       IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN m.k2o        IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN m.p2o5       IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN m.mno        IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.la_ppm     IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.ce_ppm     IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.nd_ppm     IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.sm_ppm     IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.nb_ppm     IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.zr_ppm     IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.sr_ppm     IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.ba_ppm     IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.th_ppm     IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN t.y_ppm      IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN i.sr87_sr86  IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN i.epsilon_nd IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN i.epsilon_hf IS NOT NULL THEN 1 ELSE 0 END +
              CASE WHEN i.pb206_pb204 IS NOT NULL THEN 1 ELSE 0 END
            ) AS completeness
        FROM sample_canonical_ids s
        LEFT JOIN major   m ON s.major_id = m.major_id
        LEFT JOIN trace   t ON s.trace_id = t.trace_id
        LEFT JOIN isotope i ON s.iso_id   = i.iso_id
        WHERE s.latitude  IS NOT NULL
          AND s.longitude IS NOT NULL
    """)
    rows = cursor.fetchall()
    col_names = [d[0] for d in cursor.description]

    # 转为 dict 列表，便于操作
    records = [dict(zip(col_names, r)) for r in rows]
    all_ids = [r['sample_id'] for r in records]
    id_to_idx = {r['sample_id']: i for i, r in enumerate(records)}

    # ------------------------------------------------------------------
    # 按空间格分桶，成对比较
    # ------------------------------------------------------------------
    MAJOR_COLS  = ['sio2r', 'tio2r', 'al2o3r', 'mgor', 'caor', 'na2or', 'k2or']
    TRACE_COLS  = ['lar', 'nbr', 'zrr', 'srr_ppm', 'thr']
    ISO_COLS    = ['sr_iso', 'end_iso', 'pb6_iso']

    parent = list(range(len(records)))

    spatial_buckets = defaultdict(list)
    for i, rec in enumerate(records):
        key = (rec['lat2'], rec['lon2'])
        spatial_buckets[key].append(i)

    group_counter = 0
    for indices in spatial_buckets.values():
        if len(indices) < 2:
            continue
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                ia, ib = indices[a], indices[b]
                r1, r2 = records[ia], records[ib]
                # 同文献内重复不做跨文献去重
                if r1['canonical_ref_id'] == r2['canonical_ref_id']:
                    continue
                # 多维投票（各维独立，NULL 跳过）
                match_major = _same_dim(r1, r2, MAJOR_COLS, min_compared=2)
                match_trace = _same_dim(r1, r2, TRACE_COLS, min_compared=1)
                match_iso   = _same_dim(r1, r2, ISO_COLS,   min_compared=1)
                # 空间格已匹配，额外任意一维匹配即判重
                if match_major or match_trace or match_iso:
                    _union(parent, ia, ib)

    # ------------------------------------------------------------------
    # 按 union-find 分组，每组保留完整度最高的样本
    # ------------------------------------------------------------------
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[_find(parent, i)].append(i)

    kept_set   = set()   # sample_id → 保留
    removed    = {}      # sample_id → group_id 字符串
    dup_map    = {}      # sample_id → group_id 字符串（含被保留的）

    for root, idxs in groups.items():
        if len(idxs) < 2:
            continue  # 单独样本，不涉及重复
        group_counter += 1
        gid = f"G{group_counter:06d}"
        grp_records = sorted(
            [records[i] for i in idxs],
            key=lambda r: -r['completeness']
        )
        # 最完整的保留，其余标记
        kept_set.add(grp_records[0]['sample_id'])
        for rec in grp_records[1:]:
            removed[rec['sample_id']] = gid
        for rec in grp_records:
            dup_map[rec['sample_id']] = gid

    # ------------------------------------------------------------------
    # 写入决策表
    # ------------------------------------------------------------------
    completeness_map = {r['sample_id']: r['completeness'] for r in records}
    to_insert = []
    for sid in all_ids:
        is_kept = 0 if sid in removed else 1
        gid     = dup_map.get(sid)
        comp    = completeness_map[sid]
        to_insert.append((sid, is_kept, gid, comp))

    cursor.executemany(
        "INSERT INTO dedup_decisions (sample_id, is_kept, dup_group_id, completeness) "
        "VALUES (?,?,?,?)",
        to_insert
    )
    conn.commit()

    n_removed = sum(1 for _, k, _, _ in to_insert if k == 0)
    print(f"  ✓ 共处理 {len(to_insert)} 条有坐标样本")
    print(f"  ✓ 识别重复组 {group_counter} 个")
    print(f"  ✓ 标记去重 {n_removed} 条，保留 {len(to_insert) - n_removed} 条")


# =============================================================================
# 视图创建
# =============================================================================

def create_views(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("创建研究子数据集视图")
    print("=" * 80)
    print(f"\n  年龄有效性规则: {AGE_CONDITION}")

    # =========================================================================
    # 视图 1: 完整有效数据视图（坐标+任一主量元素非空）
    # =========================================================================
    print("\n[1/7] 创建视图: v_valid_samples（完整有效数据）")
    cursor.execute("DROP VIEW IF EXISTS v_valid_samples")
    cursor.execute(f"""
        CREATE VIEW v_valid_samples AS
        SELECT
            s.sample_id, s.sample_name, s.ref_id, s.canonical_ref_id, r.author, r.data_source, r.year,
            s.method_id, s.canonical_method_id, s.age_id, s.canonical_age_id,
            s.latitude, s.longitude, s.rock_name, s.protolith,
            rg.rock_group, rg.rock_origin, rg.rock_facies,
            m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao, m.na2o, m.k2o, m.p2o5, m.mno, m.loi,
            m.fe2o3, m.fe2o3_tot, m.feo_tot, m.cr2o3, m.nio, m.sro, m.h2o_plus, m.h2o_minus, m.h2o_tot,
            m.co2, m.so3, m.bao, m.caco3, m.mgco3,
            t.la_ppm, t.ce_ppm, t.nd_ppm, t.sm_ppm, t.eu_ppm, t.gd_ppm, t.dy_ppm, t.er_ppm, t.yb_ppm, t.lu_ppm,
            t.nb_ppm, t.ta_ppm, t.zr_ppm, t.hf_ppm, t.th_ppm, t.u_ppm,
            t.rb_ppm, t.sr_ppm, t.ba_ppm, t.y_ppm, t.v_ppm, t.cr_ppm, t.co_ppm, t.ni_ppm, t.cu_ppm, t.zn_ppm,
            t.ga_ppm, t.pb_ppm, t.sc_ppm, t.li_ppm, t.be_ppm, t.b_ppm, t.ag_ppm, t.cd_ppm, t.sn_ppm, t.sb_ppm,
            t.w_ppm, t.mo_ppm, t.bi_ppm, t.se_ppm, t.te_ppm,
            i.sr87_sr86, i.epsilon_nd, i.epsilon_hf, i.pb206_pb204, i.pb207_pb204, i.pb208_pb204,
            c.mg_number, c.asi, c.mali, c.maficity, c.cia, c.wip, c.spar, c.qtzindex, c.rock_type,
            a.age, a.age_min, a.age_max, a.time_period, a.age_method
        FROM sample_canonical_ids s
        JOIN major m ON s.major_id = m.major_id
        LEFT JOIN trace   t  ON s.trace_id  = t.trace_id
        LEFT JOIN isotope i  ON s.iso_id    = i.iso_id
        LEFT JOIN computed c ON s.comp_id   = c.comp_id
        LEFT JOIN reference r ON s.canonical_ref_id = r.ref_id
        LEFT JOIN rockgroup rg ON s.rgroup_id = rg.rgroup_id
        LEFT JOIN age a ON s.canonical_age_id = a.age_id
        WHERE s.latitude  IS NOT NULL
          AND s.longitude IS NOT NULL
          AND COALESCE(m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao, m.na2o, m.k2o) IS NOT NULL
          AND {AGE_CONDITION}
    """)
    print("  ✓ 创建成功")

    # =========================================================================
    # 视图 2: 空间分析视图（有坐标，不要求主量元素完整）
    # =========================================================================
    print("\n[2/7] 创建视图: v_spatial_samples（空间分析用）")
    cursor.execute("DROP VIEW IF EXISTS v_spatial_samples")
    cursor.execute(f"""
        CREATE VIEW v_spatial_samples AS
        SELECT
            s.sample_id,
            s.sample_name,
            s.ref_id,
            s.canonical_ref_id,
            r.author,
            r.data_source,
            r.year,
            s.method_id,
            s.canonical_method_id,
            s.age_id,
            s.canonical_age_id,
            s.latitude,
            s.longitude,
            s.loc_prec,
            s.rock_name,
            s.protolith,
            rg.rock_group,
            rg.rock_origin,
            m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao,
            m.na2o, m.k2o, m.p2o5, m.loi,
            t.la_ppm, t.ce_ppm, t.nd_ppm, t.sm_ppm, t.eu_ppm,
            t.nb_ppm, t.zr_ppm, t.th_ppm, t.y_ppm,
            t.sr_ppm, t.ba_ppm, t.v_ppm, t.cr_ppm, t.ni_ppm,
            a.age, a.time_period
        FROM sample_canonical_ids s
        LEFT JOIN major   m  ON s.major_id  = m.major_id
        LEFT JOIN trace   t  ON s.trace_id  = t.trace_id
        LEFT JOIN reference r ON s.canonical_ref_id = r.ref_id
        LEFT JOIN rockgroup rg ON s.rgroup_id = rg.rgroup_id
        LEFT JOIN age a ON s.canonical_age_id = a.age_id
        WHERE s.latitude  IS NOT NULL
          AND s.longitude IS NOT NULL
          AND {AGE_CONDITION}
    """)
    print("  ✓ 创建成功")

    # =========================================================================
    # 视图 3: 地球化学分析视图（有主量+微量，不要求坐标）
    # =========================================================================
    print("\n[3/7] 创建视图: v_geochem_complete（地球化学完整数据）")
    cursor.execute("DROP VIEW IF EXISTS v_geochem_complete")
    cursor.execute(f"""
        CREATE VIEW v_geochem_complete AS
        SELECT
            s.sample_id,
            s.sample_name,
            s.ref_id,
            s.canonical_ref_id,
            r.author,
            r.data_source,
            r.year,
            s.method_id,
            s.canonical_method_id,
            s.age_id,
            s.canonical_age_id,
            s.latitude,
            s.longitude,
            s.rock_name,
            s.protolith,
            rg.rock_group,
            rg.rock_origin,
            m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao,
            m.na2o, m.k2o, m.p2o5, m.mno, m.loi,
            m.fe2o3, m.fe2o3_tot, m.feo_tot, m.cr2o3, m.nio,
            m.sro, m.h2o_plus, m.h2o_minus, m.h2o_tot,
            m.co2, m.so3, m.bao, m.caco3, m.mgco3,
            t.f_ppm, t.cl_ppm, t.br_ppm, t.i_ppm, t.h_ppm,
            t.c_ppm, t.n_ppm, t.p_ppm, t.s_ppm,
            t.al_ppm, t.as_ppm, t.ag_ppm, t.au_ppm, t.b_ppm,
            t.ba_ppm, t.be_ppm, t.bi_ppm, t.ca_ppm, t.cd_ppm,
            t.ce_ppm, t.co_ppm, t.cr_ppm, t.cs_ppm, t.cu_ppm,
            t.dy_ppm, t.er_ppm, t.eu_ppm, t.fe_ppm, t.ga_ppm,
            t.gd_ppm, t.ge_ppm, t.hf_ppm, t.hg_ppm, t.ho_ppm,
            t.in_ppm, t.ir_ppm, t.k_ppm, t.la_ppm, t.li_ppm,
            t.lu_ppm, t.mg_ppm, t.mn_ppm, t.mo_ppm, t.na_ppm,
            t.nd_ppm, t.ni_ppm, t.nb_ppm, t.os_ppm, t.pa_ppm,
            t.pb_ppm, t.pd_ppm, t.pm_ppm, t.pr_ppm, t.pt_ppm,
            t.rb_ppm, t.re_ppm, t.rh_ppm, t.ru_ppm, t.sb_ppm,
            t.sc_ppm, t.se_ppm, t.si_ppm, t.sm_ppm, t.sn_ppm,
            t.sr_ppm, t.ta_ppm, t.tb_ppm, t.te_ppm, t.th_ppm,
            t.ti_ppm, t.tl_ppm, t.tm_ppm, t.w_ppm,  t.v_ppm,
            t.u_ppm,  t.y_ppm,  t.yb_ppm, t.zn_ppm, t.zr_ppm,
            i.sr87_sr86, i.epsilon_nd, i.epsilon_hf,
            i.pb206_pb204, i.pb207_pb204, i.pb208_pb204,
            c.mg_number, c.asi, c.mali, c.maficity,
            c.cia, c.wip, c.spar, c.qtzindex, c.rock_type,
            a.age, a.age_min, a.age_max, a.time_period, a.age_method
        FROM sample_canonical_ids s
        JOIN major m ON s.major_id = m.major_id
        LEFT JOIN trace   t  ON s.trace_id  = t.trace_id
        LEFT JOIN isotope i  ON s.iso_id    = i.iso_id
        LEFT JOIN computed c ON s.comp_id   = c.comp_id
        LEFT JOIN reference r ON s.canonical_ref_id = r.ref_id
        LEFT JOIN rockgroup rg ON s.rgroup_id = rg.rgroup_id
        LEFT JOIN age a ON s.canonical_age_id = a.age_id
        WHERE m.sio2  IS NOT NULL
          AND m.tio2  IS NOT NULL
          AND m.al2o3 IS NOT NULL
          AND m.mgo   IS NOT NULL
          AND m.feo   IS NOT NULL
          AND m.cao   IS NOT NULL
          AND m.na2o  IS NOT NULL
          AND m.k2o   IS NOT NULL
          AND {AGE_CONDITION}
    """)
    print("  ✓ 创建成功")

    # =========================================================================
    # 视图 4: 多维去重视图（引用 dedup_decisions，推荐）
    # =========================================================================
    print("\n[4/7] 创建视图: v_non_duplicate（多维去重，推荐）")
    cursor.execute("DROP VIEW IF EXISTS v_non_duplicate")
    cursor.execute(f"""
        CREATE VIEW v_non_duplicate AS
        SELECT
            s.sample_id,
            s.sample_name,
            s.ref_id,
            s.canonical_ref_id,
            r.author,
            r.data_source,
            r.year,
            s.method_id,
            s.canonical_method_id,
            s.age_id,
            s.canonical_age_id,
            d.dup_group_id,
            d.completeness,
            s.latitude,
            s.longitude,
            s.rock_name,
            s.protolith,
            rg.rock_group,
            rg.rock_origin,
            m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao,
            m.na2o, m.k2o, m.p2o5, m.mno, m.loi,
            t.la_ppm, t.ce_ppm, t.nd_ppm, t.sm_ppm, t.eu_ppm,
            t.nb_ppm, t.zr_ppm, t.th_ppm, t.y_ppm,
            t.sr_ppm, t.ba_ppm, t.v_ppm, t.cr_ppm, t.ni_ppm,
            i.sr87_sr86, i.epsilon_nd, i.epsilon_hf,
            i.pb206_pb204, i.pb207_pb204, i.pb208_pb204,
            c.mg_number, c.asi, c.mali, c.maficity,
            c.cia, c.wip, c.spar, c.qtzindex, c.rock_type,
            a.age, a.age_min, a.age_max, a.time_period, a.age_method
        FROM sample_canonical_ids s
        JOIN dedup_decisions d ON s.sample_id = d.sample_id
        LEFT JOIN major   m  ON s.major_id  = m.major_id
        LEFT JOIN trace   t  ON s.trace_id  = t.trace_id
        LEFT JOIN isotope i  ON s.iso_id    = i.iso_id
        LEFT JOIN computed c ON s.comp_id   = c.comp_id
        LEFT JOIN reference r ON s.canonical_ref_id = r.ref_id
        LEFT JOIN rockgroup rg ON s.rgroup_id = rg.rgroup_id
        LEFT JOIN age a ON s.canonical_age_id = a.age_id
        WHERE d.is_kept = 1
          AND {AGE_CONDITION}
    """)
    print("  ✓ 创建成功")

    # =========================================================================
    # 视图 5: 重复组审查视图（人工核查用）
    # =========================================================================
    print("\n[5/7] 创建视图: v_duplicate_review（重复组审查）")
    cursor.execute("DROP VIEW IF EXISTS v_duplicate_review")
    cursor.execute("""
        CREATE VIEW v_duplicate_review AS
        SELECT
            d.dup_group_id,
            d.is_kept,
            d.completeness,
            s.sample_id,
            s.sample_name,
            s.ref_id,
            s.canonical_ref_id,
            r.author,
            r.data_source,
            r.year,
            s.latitude,
            s.longitude,
            m.sio2, m.tio2, m.al2o3, m.mgo,
            t.la_ppm, t.nb_ppm, t.zr_ppm,
            i.sr87_sr86, i.epsilon_nd
        FROM dedup_decisions d
        JOIN sample_canonical_ids s ON s.sample_id = d.sample_id
        LEFT JOIN major   m  ON s.major_id = m.major_id
        LEFT JOIN trace   t  ON s.trace_id = t.trace_id
        LEFT JOIN isotope i  ON s.iso_id   = i.iso_id
        LEFT JOIN reference r ON s.canonical_ref_id = r.ref_id
        WHERE d.dup_group_id IS NOT NULL
        ORDER BY d.dup_group_id, d.is_kept DESC
    """)
    print("  ✓ 创建成功")

    # =========================================================================
    # 视图 6: 高质量综合数据集（完整+去重+年龄有效）
    # =========================================================================
    print("\n[6/7] 创建视图: v_high_quality_dataset（高质量综合数据集）")
    cursor.execute("DROP VIEW IF EXISTS v_high_quality_dataset")
    cursor.execute(f"""
        CREATE VIEW v_high_quality_dataset AS
        SELECT
            s.sample_id,
            s.sample_name,
            s.ref_id,
            s.canonical_ref_id,
            r.author,
            r.data_source,
            r.year,
            s.method_id,
            s.canonical_method_id,
            s.age_id,
            s.canonical_age_id,
            d.dup_group_id,
            d.completeness,
            s.latitude,
            s.longitude,
            s.rock_name,
            s.protolith,
            s.material,
            rg.rock_group,
            rg.rock_origin,
            rg.rock_facies,
            m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao,
            m.na2o, m.k2o, m.p2o5, m.mno, m.loi,
            m.fe2o3, m.fe2o3_tot, m.feo_tot, m.cr2o3, m.nio,
            m.sro, m.h2o_plus, m.h2o_minus, m.h2o_tot,
            m.co2, m.so3, m.bao, m.caco3, m.mgco3,
            t.la_ppm, t.ce_ppm, t.nd_ppm, t.sm_ppm, t.eu_ppm,
            t.gd_ppm, t.dy_ppm, t.er_ppm, t.yb_ppm, t.lu_ppm,
            t.nb_ppm, t.ta_ppm, t.zr_ppm, t.hf_ppm, t.th_ppm, t.u_ppm,
            t.rb_ppm, t.sr_ppm, t.ba_ppm, t.y_ppm, t.v_ppm,
            t.cr_ppm, t.co_ppm, t.ni_ppm, t.cu_ppm, t.zn_ppm,
            t.ga_ppm, t.pb_ppm, t.sc_ppm, t.li_ppm, t.be_ppm,
            t.b_ppm, t.ag_ppm, t.cd_ppm, t.sn_ppm, t.sb_ppm,
            t.w_ppm, t.mo_ppm, t.bi_ppm, t.se_ppm, t.te_ppm,
            i.sr87_sr86, i.epsilon_nd, i.epsilon_hf,
            i.pb206_pb204, i.pb207_pb204, i.pb208_pb204,
            c.mg_number, c.asi, c.mali, c.maficity,
            c.cia, c.wip, c.spar, c.qtzindex, c.rock_type,
            a.age, a.age_min, a.age_max, a.time_period, a.age_method
        FROM sample_canonical_ids s
        JOIN major m ON s.major_id = m.major_id
        JOIN dedup_decisions d ON s.sample_id = d.sample_id
        LEFT JOIN trace   t  ON s.trace_id  = t.trace_id
        LEFT JOIN isotope i  ON s.iso_id    = i.iso_id
        LEFT JOIN computed c ON s.comp_id   = c.comp_id
        LEFT JOIN reference r ON s.canonical_ref_id = r.ref_id
        LEFT JOIN rockgroup rg ON s.rgroup_id = rg.rgroup_id
        LEFT JOIN age a ON s.canonical_age_id = a.age_id
        WHERE s.latitude  IS NOT NULL
          AND s.longitude IS NOT NULL
          AND m.sio2  IS NOT NULL
          AND m.tio2  IS NOT NULL
          AND m.al2o3 IS NOT NULL
          AND m.feo   IS NOT NULL
          AND m.mgo   IS NOT NULL
          AND m.cao   IS NOT NULL
          AND m.na2o  IS NOT NULL
          AND m.k2o   IS NOT NULL
          AND d.is_kept = 1
          AND {AGE_CONDITION}
    """)
    print("  ✓ 创建成功")

# =========================================================================
    # 视图 7: 无效数据视图（数据质量检查）
    # =========================================================================
    print("\n[7/7] 创建视图: v_invalid_records（无效数据检查）")
    cursor.execute("DROP VIEW IF EXISTS v_invalid_records")
    cursor.execute("""
        CREATE VIEW v_invalid_records AS
        SELECT
            s.sample_id, s.sample_name, s.ref_id, s.canonical_ref_id, r.author, r.data_source,
            s.method_id, s.canonical_method_id, s.age_id, s.canonical_age_id,
            s.latitude, s.longitude, s.rock_name, rg.rock_group,
            m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao, m.na2o, m.k2o, m.p2o5, m.loi,
            a.age,
            CASE
                WHEN a.age IS NOT NULL AND a.age <= 0
                    THEN '年龄无效（age <= 0）'
                WHEN a.age IS NOT NULL AND a.age > 10000
                    THEN '年龄超限（age > 10000 Ma）'
                WHEN s.latitude IS NULL AND s.longitude IS NULL
                     AND COALESCE(m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao, m.na2o, m.k2o) IS NULL
                    THEN '坐标和主量元素都缺失'
                WHEN s.latitude IS NOT NULL AND s.longitude IS NOT NULL
                     AND COALESCE(m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao, m.na2o, m.k2o) IS NULL
                    THEN '有坐标但完全缺少主量元素'
                WHEN (s.latitude IS NULL OR s.longitude IS NULL)
                     AND COALESCE(m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao, m.na2o, m.k2o) IS NOT NULL
                    THEN '缺少坐标但有主量元素'
                ELSE '其他'
            END AS invalid_reason
        FROM sample_canonical_ids s
        LEFT JOIN major   m  ON s.major_id  = m.major_id
        LEFT JOIN reference r ON s.canonical_ref_id = r.ref_id
        LEFT JOIN rockgroup rg ON s.rgroup_id = rg.rgroup_id
        LEFT JOIN age a ON s.canonical_age_id = a.age_id
        WHERE s.latitude  IS NULL
           OR s.longitude IS NULL
           OR COALESCE(m.sio2, m.tio2, m.al2o3, m.feo, m.mgo, m.cao, m.na2o, m.k2o) IS NULL
           OR (a.age IS NOT NULL AND (a.age <= 0 OR a.age > 10000))
    """)
    print("  ✓ 创建成功")

    conn.commit()
    print("\n" + "=" * 80)
    print("所有视图创建完成！")
    print("=" * 80)


# =============================================================================
# 统计信息
# =============================================================================

def show_view_statistics(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("视图数据统计")
    print("=" * 80)

    views = [
        ('sample_canonical_ids',   '属性映射样本视图'),
        ('v_valid_samples',       '完整有效数据'),
        ('v_spatial_samples',     '空间分析数据'),
        ('v_geochem_complete',    '地球化学完整数据'),
        ('v_non_duplicate',       '多维去重数据（推荐）'),
        ('v_duplicate_review',    '重复组审查'),
        ('v_high_quality_dataset','高质量综合数据集'),
        ('v_invalid_records',     '无效数据'),
    ]

    cursor.execute("SELECT COUNT(*) FROM sample")
    total = cursor.fetchone()[0]

    print(f"\n{'视图名称':<30} {'说明':<20} {'记录数':>10}")
    print("-" * 62)

    for view_name, description in views:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {view_name}")
            count = cursor.fetchone()[0]
            print(f"{view_name:<30} {description:<20} {count:>10,}")
        except Exception as e:
            print(f"{view_name:<30} {description:<20} {'ERROR':>10}  ({e})")

    print("-" * 62)
    print(f"{'sample（总计）':<30} {'':<20} {total:>10,}")

    # 额外：年龄规则影响统计
    print("\n  --- 年龄规则过滤情况 ---")
    try:
        cursor.execute("SELECT COUNT(*) FROM age WHERE age IS NOT NULL AND age <= 0")
        c1 = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM age WHERE age IS NOT NULL AND age > 10000")
        c2 = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM age WHERE age IS NULL")
        c_null = cursor.fetchone()[0]
        print(f"  age <= 0（无效）       : {c1:,} 条")
        print(f"  age > 10000 Ma（超限）  : {c2:,} 条")
        print(f"  age IS NULL（未知，保留）: {c_null:,} 条")
    except Exception as e:
        print(f"  （无法查询 age 表: {e}）")


def _get_view_columns(cursor: sqlite3.Cursor, view_name: str) -> list:
    """返回给定视图的字段名称列表。"""
    try:
        cursor.execute(f"SELECT * FROM {view_name} WHERE 0")
        return [d[0] for d in cursor.description]
    except Exception:
        return []


# =============================================================================
# 文档生成
# =============================================================================

def create_view_documentation(conn: sqlite3.Connection, output_file: str) -> None:
    cursor = conn.cursor()

    views_meta = [
        ('v_valid_samples',        '完整有效数据',       '坐标+主量元素非空，适合多数研究'),
        ('v_spatial_samples',      '空间分析数据',       '所有有坐标的样本，适合空间分布分析'),
        ('v_geochem_complete',     '地球化学完整数据',   '主量+微量完整，适合地球化学分类判别'),
        ('v_non_duplicate',        '多维去重（推荐）',   '跨文献重复已去除，保留完整度最高的样本'),
        ('v_duplicate_review',     '重复组审查',         '列出所有重复组，含保留/丢弃标记，供人工核查'),
        ('v_high_quality_dataset', '高质量综合数据集',   '完整+去重+年龄有效，适合机器学习/统计分析'),
        ('v_invalid_records',      '无效数据',           '坐标/主量/年龄任一无效，不应用于研究'),
    ]

    doc = f"""# 研究子数据集视图说明

## 年龄有效性规则

| 条件 | 处理 |
|------|------|
| `age IS NULL` | 保留（未知年龄，不排除） |
| `0 < age <= 10000` | 保留（正常地质年龄范围，单位 Ma） |
| `age <= 0` | **排除**（物理上无意义） |
| `age > 10000` | **排除**（超过地球年龄上限） |

SQL 条件统一写作：`{AGE_CONDITION}`

## 视图列表

| 视图名称 | 说明 | 适用研究 | 记录数 |
|----------|------|----------|--------|
"""

    for view_name, description, usage in views_meta:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {view_name}")
            count = cursor.fetchone()[0]
            doc += f"| `{view_name}` | {description} | {usage} | {count:,} |\n"
        except Exception:
            doc += f"| `{view_name}` | {description} | {usage} | ERROR |\n"

    doc += """
## 视图字段详情

以下列出每个视图所包含的字段名，便于了解输出结构与可用特征。
"""

    for view_name, description, usage in views_meta:
        columns = _get_view_columns(cursor, view_name)
        if columns:
            doc += f"\n### `{view_name}`：{description}\n\n"
            doc += "```\n"
            doc += "\n".join(columns)
            doc += "\n```\n"
        else:
            doc += f"\n### `{view_name}`：{description}\n\n"
            doc += "无法获取字段列表。可能是视图尚未创建或查询失败。\n"

    doc += """
## 重复性检验说明

### 策略：多维加权指纹

传统方案仅用 SiO₂+TiO₂ 等少数字段判重，存在以下问题：

- **误杀**：两个真实不同的样本某元素恰好相等；
- **漏判**：两篇论文共用数据，一篇只发表主量、另一篇只发表微量，
  对比同一维度时字段均为 NULL，导致无法识别；
- **NULL 惩罚**：传统 NOT IN 子查询中，NULL 字段直接令记录被排除。

新方案流程：

1. **空间格聚类**：`ROUND(lat, 2) × ROUND(lon, 2)` 为粒度（约 ±1 km），
   同格内样本进入成对比较。
2. **多维独立打分**：主量（SiO₂、TiO₂、Al₂O₃、MgO、CaO、Na₂O、K₂O）、
   微量（La、Nb、Zr、Sr、Th）、同位素（⁸⁷Sr/⁸⁶Sr、εNd、Pb 比值）
   三个维度分别判断。
3. **NULL 跳过**：某维度双方任一为 NULL，该维度不参与投票，不惩罚得分。
4. **判重条件**：空间格已匹配 + 至少 1 个额外维度匹配 → 标记为重复对。
5. **同文献保留**：同一 `ref_id` 内的重复属采样设计，不参与跨文献去重。
6. **组内保留最优**：按非空字段总数（completeness）降序，保留信息最完整的样本。

### 审查重复组

```sql
-- 查看某重复组的所有样本
SELECT * FROM v_duplicate_review WHERE dup_group_id = 'G000001';

-- 统计各组大小
SELECT dup_group_id, COUNT(*) AS n_samples,
       SUM(is_kept) AS n_kept
FROM v_duplicate_review
GROUP BY dup_group_id;
```

## 使用示例

### SQL 查询
```sql
-- 基础有效数据
SELECT * FROM v_valid_samples WHERE rock_group = 'Basalt';

-- 空间分布（带年龄筛选）
SELECT latitude, longitude, rock_name, age
FROM v_spatial_samples
WHERE age BETWEEN 0 AND 200;

-- 地球化学判别图
SELECT sio2, (na2o + k2o) AS total_alkali, rock_group
FROM v_geochem_complete;

-- 推荐：去重后高质量数据集
SELECT * FROM v_high_quality_dataset WHERE sio2 > 45 AND sio2 < 55;
```

### Python 使用
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/geochem.db')

# 高质量无重复数据集
df = pd.read_sql('SELECT * FROM v_high_quality_dataset', conn)

# 查看重复组
dup = pd.read_sql('SELECT * FROM v_duplicate_review', conn)
```

## 视图刷新

原始数据更新后，重新运行脚本以刷新视图及去重决策表：

```bash
python scripts/create_research_views.py
```
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(doc)

    print(f"\n  文档已保存至: {output_file}")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    print("\n研究子数据集视图创建工具")
    print("=" * 80)

    if not DB_PATH.exists():
        print(f"错误：数据库文件不存在: {DB_PATH}")
        return

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    conn = connect()
    try:
        # 1. 构建属性层 canonical mapping
        build_canonical_id_maps(conn)

        # 2. 构建去重决策表（基于 canonical ref_id）
        build_dedup_table(conn)

        # 3. 创建所有视图
        create_views(conn)

        # 4. 统计信息
        show_view_statistics(conn)

        # 5. 生成文档
        doc_file = str(DOCS_DIR / "research_views_documentation.md")
        create_view_documentation(conn, doc_file)

        print("\n" + "=" * 80)
        print("使用说明")
        print("=" * 80)
        print("""
推荐研究用途对照：

  空间分析          →  v_spatial_samples
  地球化学分类/图解  →  v_geochem_complete
  去重数据集        →  v_non_duplicate        ← 推荐
  全量高质量分析    →  v_high_quality_dataset  ← 最严格
  重复组人工核查    →  v_duplicate_review
  数据质量检查      →  v_invalid_records

年龄规则（所有视图统一）：
  保留 age IS NULL
  保留 0 < age <= 10000 Ma
  排除 age <= 0 或 age > 10000
""")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
