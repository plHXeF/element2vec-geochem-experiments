# 研究子数据集视图说明

## 年龄有效性规则

| 条件 | 处理 |
|------|------|
| `age IS NULL` | 保留（未知年龄，不排除） |
| `0 < age <= 10000` | 保留（正常地质年龄范围，单位 Ma） |
| `age <= 0` | **排除**（物理上无意义） |
| `age > 10000` | **排除**（超过地球年龄上限） |

SQL 条件统一写作：`(a.age IS NULL OR (a.age > 0 AND a.age <= 10000))`

## 视图列表

| 视图名称 | 说明 | 适用研究 | 记录数 |
|----------|------|----------|--------|
| `v_valid_samples` | 完整有效数据 | 坐标+主量元素非空，适合多数研究 | 682,801 |
| `v_spatial_samples` | 空间分析数据 | 所有有坐标的样本，适合空间分布分析 | 1,005,479 |
| `v_geochem_complete` | 地球化学完整数据 | 主量+微量完整，适合地球化学分类判别 | 293,343 |
| `v_non_duplicate` | 多维去重（推荐） | 跨文献重复已去除，保留完整度最高的样本 | 927,411 |
| `v_duplicate_review` | 重复组审查 | 列出所有重复组，含保留/丢弃标记，供人工核查 | 114,610 |
| `v_high_quality_dataset` | 高质量综合数据集 | 完整+去重+年龄有效，适合机器学习/统计分析 | 270,241 |
| `v_invalid_records` | 无效数据 | 坐标/主量/年龄任一无效，不应用于研究 | 339,291 |

## 视图字段详情

以下列出每个视图所包含的字段名，便于了解输出结构与可用特征。

### `v_valid_samples`：完整有效数据

```
sample_id
sample_name
ref_id
canonical_ref_id
author
data_source
year
method_id
canonical_method_id
age_id
canonical_age_id
latitude
longitude
rock_name
protolith
rock_group
rock_origin
rock_facies
sio2
tio2
al2o3
feo
mgo
cao
na2o
k2o
p2o5
mno
loi
fe2o3
fe2o3_tot
feo_tot
cr2o3
nio
sro
h2o_plus
h2o_minus
h2o_tot
co2
so3
bao
caco3
mgco3
la_ppm
ce_ppm
nd_ppm
sm_ppm
eu_ppm
gd_ppm
dy_ppm
er_ppm
yb_ppm
lu_ppm
nb_ppm
ta_ppm
zr_ppm
hf_ppm
th_ppm
u_ppm
rb_ppm
sr_ppm
ba_ppm
y_ppm
v_ppm
cr_ppm
co_ppm
ni_ppm
cu_ppm
zn_ppm
ga_ppm
pb_ppm
sc_ppm
li_ppm
be_ppm
b_ppm
ag_ppm
cd_ppm
sn_ppm
sb_ppm
w_ppm
mo_ppm
bi_ppm
se_ppm
te_ppm
sr87_sr86
epsilon_nd
epsilon_hf
pb206_pb204
pb207_pb204
pb208_pb204
mg_number
asi
mali
maficity
cia
wip
spar
qtzindex
rock_type
age
age_min
age_max
time_period
age_method
```

### `v_spatial_samples`：空间分析数据

```
sample_id
sample_name
ref_id
canonical_ref_id
author
data_source
year
method_id
canonical_method_id
age_id
canonical_age_id
latitude
longitude
loc_prec
rock_name
protolith
rock_group
rock_origin
sio2
tio2
al2o3
feo
mgo
cao
na2o
k2o
p2o5
loi
la_ppm
ce_ppm
nd_ppm
sm_ppm
eu_ppm
nb_ppm
zr_ppm
th_ppm
y_ppm
sr_ppm
ba_ppm
v_ppm
cr_ppm
ni_ppm
age
time_period
```

### `v_geochem_complete`：地球化学完整数据

```
sample_id
sample_name
ref_id
canonical_ref_id
author
data_source
year
method_id
canonical_method_id
age_id
canonical_age_id
latitude
longitude
rock_name
protolith
rock_group
rock_origin
sio2
tio2
al2o3
feo
mgo
cao
na2o
k2o
p2o5
mno
loi
fe2o3
fe2o3_tot
feo_tot
cr2o3
nio
sro
h2o_plus
h2o_minus
h2o_tot
co2
so3
bao
caco3
mgco3
f_ppm
cl_ppm
br_ppm
i_ppm
h_ppm
c_ppm
n_ppm
p_ppm
s_ppm
al_ppm
as_ppm
ag_ppm
au_ppm
b_ppm
ba_ppm
be_ppm
bi_ppm
ca_ppm
cd_ppm
ce_ppm
co_ppm
cr_ppm
cs_ppm
cu_ppm
dy_ppm
er_ppm
eu_ppm
fe_ppm
ga_ppm
gd_ppm
ge_ppm
hf_ppm
hg_ppm
ho_ppm
in_ppm
ir_ppm
k_ppm
la_ppm
li_ppm
lu_ppm
mg_ppm
mn_ppm
mo_ppm
na_ppm
nd_ppm
ni_ppm
nb_ppm
os_ppm
pa_ppm
pb_ppm
pd_ppm
pm_ppm
pr_ppm
pt_ppm
rb_ppm
re_ppm
rh_ppm
ru_ppm
sb_ppm
sc_ppm
se_ppm
si_ppm
sm_ppm
sn_ppm
sr_ppm
ta_ppm
tb_ppm
te_ppm
th_ppm
ti_ppm
tl_ppm
tm_ppm
w_ppm
v_ppm
u_ppm
y_ppm
yb_ppm
zn_ppm
zr_ppm
sr87_sr86
epsilon_nd
epsilon_hf
pb206_pb204
pb207_pb204
pb208_pb204
mg_number
asi
mali
maficity
cia
wip
spar
qtzindex
rock_type
age
age_min
age_max
time_period
age_method
```

### `v_non_duplicate`：多维去重（推荐）

```
sample_id
sample_name
ref_id
canonical_ref_id
author
data_source
year
method_id
canonical_method_id
age_id
canonical_age_id
dup_group_id
completeness
latitude
longitude
rock_name
protolith
rock_group
rock_origin
sio2
tio2
al2o3
feo
mgo
cao
na2o
k2o
p2o5
mno
loi
la_ppm
ce_ppm
nd_ppm
sm_ppm
eu_ppm
nb_ppm
zr_ppm
th_ppm
y_ppm
sr_ppm
ba_ppm
v_ppm
cr_ppm
ni_ppm
sr87_sr86
epsilon_nd
epsilon_hf
pb206_pb204
pb207_pb204
pb208_pb204
mg_number
asi
mali
maficity
cia
wip
spar
qtzindex
rock_type
age
age_min
age_max
time_period
age_method
```

### `v_duplicate_review`：重复组审查

```
dup_group_id
is_kept
completeness
sample_id
sample_name
ref_id
canonical_ref_id
author
data_source
year
latitude
longitude
sio2
tio2
al2o3
mgo
la_ppm
nb_ppm
zr_ppm
sr87_sr86
epsilon_nd
```

### `v_high_quality_dataset`：高质量综合数据集

```
sample_id
sample_name
ref_id
canonical_ref_id
author
data_source
year
method_id
canonical_method_id
age_id
canonical_age_id
dup_group_id
completeness
latitude
longitude
rock_name
protolith
material
rock_group
rock_origin
rock_facies
sio2
tio2
al2o3
feo
mgo
cao
na2o
k2o
p2o5
mno
loi
fe2o3
fe2o3_tot
feo_tot
cr2o3
nio
sro
h2o_plus
h2o_minus
h2o_tot
co2
so3
bao
caco3
mgco3
la_ppm
ce_ppm
nd_ppm
sm_ppm
eu_ppm
gd_ppm
dy_ppm
er_ppm
yb_ppm
lu_ppm
nb_ppm
ta_ppm
zr_ppm
hf_ppm
th_ppm
u_ppm
rb_ppm
sr_ppm
ba_ppm
y_ppm
v_ppm
cr_ppm
co_ppm
ni_ppm
cu_ppm
zn_ppm
ga_ppm
pb_ppm
sc_ppm
li_ppm
be_ppm
b_ppm
ag_ppm
cd_ppm
sn_ppm
sb_ppm
w_ppm
mo_ppm
bi_ppm
se_ppm
te_ppm
sr87_sr86
epsilon_nd
epsilon_hf
pb206_pb204
pb207_pb204
pb208_pb204
mg_number
asi
mali
maficity
cia
wip
spar
qtzindex
rock_type
age
age_min
age_max
time_period
age_method
```

### `v_invalid_records`：无效数据

```
sample_id
sample_name
ref_id
canonical_ref_id
author
data_source
method_id
canonical_method_id
age_id
canonical_age_id
latitude
longitude
rock_name
rock_group
sio2
tio2
al2o3
feo
mgo
cao
na2o
k2o
p2o5
loi
age
invalid_reason
```

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
