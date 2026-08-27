#!/usr/bin/env python3
"""
探索 DeepStream 检测目标真实元数据 (pyservicemaker 视角)

在给定视频上跑 person 整帧检测 (nvinfer pgie, uid=1)，然后用一个
BatchMetadataOperator 探针把每一帧 / 每个检测目标上**实际暴露**的属性
全部转储出来 —— 让你看清 DeepStream 里检测到的目标数据结构到底是什么样。

三种模式:
    --dump-schema   只打印 wrapper 暴露的字段名 (dir)，不看值 (安全)
    --dump-frame N  转储前 N 帧的完整字段值 (含 rect_params 嵌套 / classifier_items)
    --full          叠加 helmet+harness_cls+vest_cls，转储二级分类器元数据
                    (classifier_items / get_n_label)

用法:
    python3 tools/explore_metadata.py --file 'tmp/Mobile Camera0676.mp4' --dump-schema
    python3 tools/explore_metadata.py --file 'tmp/Mobile Camera0676.mp4' --dump-frame 5
    python3 tools/explore_metadata.py --file 'tmp/Mobile Camera0676.mp4' --full --dump-frame 5
"""

import argparse
import os
import sys
from multiprocessing import Process
from pathlib import Path

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# INI 路径锚定逻辑与运行期共用同一份实现（部署根 = deploy/）
from server.pipeline.ini_patch import anchor_ini_config as _anchor_ini_config  # noqa: E402

DEPLOY_ROOT = ROOT / "deploy"


# ---------------------------------------------------------------------------
# 转储工具
# ---------------------------------------------------------------------------
_out = sys.stdout


def _p(*a):
    """打印并立即 flush（探针在流线程，os._exit 不 flush 缓冲）。"""
    print(*a, file=_out, flush=True)


def _dump_dir(obj, title, indent=0):
    """列出 wrapper 暴露的所有属性名（不取值，安全）。"""
    pad = "  " * indent
    names = sorted(n for n in dir(obj) if not n.startswith("__"))
    _p(f"{pad}{title}: {len(names)} attrs")
    for n in names:
        _p(f"{pad}  .{n}")


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------
class MetadataDumper(BatchMetadataOperator):
    """按已核验的字段逐一取值转储（避免 pybind11 硬崩溃的字段）。"""

    _FRAME_FIELDS = (
        "pad_index", "batch_id", "frame_number", "buffer_pts", "ntp_timestamp",
        "source_id", "source_width", "source_height",
        "pipeline_width", "pipeline_height",
    )
    _OBJ_FIELDS = (
        "unique_component_id", "class_id", "label", "confidence",
        "tracker_confidence", "object_id",
    )
    _RECT_FIELDS = ("left", "top", "width", "height", "border_width",
                    "rotation_angle", "has_bg_color")

    def __init__(self, frames: int):
        super().__init__()
        self.frames = frames
        self.done = 0

    @staticmethod
    def _g(obj, name):
        """安全读单个字段，失败返回 '<err>'（不崩溃）。"""
        try:
            return getattr(obj, name)
        except Exception:
            return "<read-error>"

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            if self.done >= self.frames:
                os._exit(0)
            self.done += 1
            self._dump_frame(frame_meta)

    # ---- 具体转储 ----
    def _dump_frame(self, fm):
        _p("=" * 70)
        _p(f"FRAME #{self.done}  (frame_meta, type={type(fm).__name__})")
        for f in self._FRAME_FIELDS:
            _p(f"  .{f} = {self._g(fm, f)!r}")
        # 帧级各元数据容器数量
        for coll in ("user_meta_items", "tensor_items",
                     "segmentation_items", "nvdsanalytics_frame_items"):
            if hasattr(fm, coll):
                try:
                    n = len(list(getattr(fm, coll)))
                except Exception:
                    n = "?"
                _p(f"  .{coll}: {n} 项")
        # object_items 是"一次性迭代器"：包装器只在直接迭代期间有效，
        # 不能 list() 后复用（会段错误/静默崩溃），必须边迭代边取值。
        n = 0
        shown = 0
        for o in fm.object_items:
            n += 1
            if shown < 6:
                shown += 1
                self._dump_object(o)
        _p(f"  -> 本帧 object_items 共 {n} 个对象 (含所有 gie-unique-id)")

    def _dump_object(self, o):
        _p(f"  - object_meta (type={type(o).__name__})")
        for f in self._OBJ_FIELDS:
            _p(f"    .{f} = {self._g(o, f)!r}")
        rp = self._g(o, "rect_params")
        if rp is not None:
            _p(f"    .rect_params (type={type(rp).__name__}):")
            for f in self._RECT_FIELDS:
                _p(f"      .{f} = {self._g(rp, f)!r}")
            # 颜色对象展开
            for cf in ("border_color", "bg_color"):
                c = self._g(rp, cf)
                if c is not None:
                    _p(f"      .{cf} = (r={self._g(c,'r')}, g={self._g(c,'g')}, "
                       f"b={self._g(c,'b')}, a={self._g(c,'a')})")
        # 分类器元数据（二级分类器挂这里）；同样直接迭代，不 list()
        if hasattr(o, "classifier_items"):
            n_clf = 0
            for clf in o.classifier_items:
                n_clf += 1
                self._dump_classifier(clf)
            _p(f"    .classifier_items: {n_clf} 个分类器元数据")

    def _dump_classifier(self, clf):
        _p(f"      + classifier_meta (type={type(clf).__name__})")
        for f in ("unique_component_id", "n_labels", "classifier_type"):
            _p(f"        .{f} = {self._g(clf, f)!r}")
        n = self._g(clf, "n_labels") or 0
        for i in range(n):
            try:
                raw = clf.get_n_label(i)
            except Exception as e:
                raw = f"<err {type(e).__name__}>"
            _p(f"        .get_n_label({i}) = {raw!r}")



# ---------------------------------------------------------------------------
# 模式: 只打印 schema
# ---------------------------------------------------------------------------
class SchemaDumper(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self.done = 0

    def handle_metadata(self, batch_meta):
        for fm in batch_meta.frame_items:
            self.done += 1
            if self.done > 2:
                os._exit(0)
            _p("=" * 60)
            _p(f"FRAME #{self.done} (type={type(fm).__name__})")
            _dump_dir(fm, "  frame_meta attrs", indent=1)
            for o in fm.object_items:
                _dump_dir(o, "  object_meta attrs", indent=1)
                _dump_dir(o.rect_params, "    rect_params attrs", indent=2)
                if hasattr(o, "classifier_items"):
                    try:
                        clfs = list(o.classifier_items)
                    except Exception:
                        clfs = []
                    if clfs:
                        _dump_dir(clfs[0], "    classifier_meta attrs", indent=2)
                        n = getattr(clfs[0], "n_labels", 0)
                        _p(f"      .n_labels = {n} ; get_n_label(0) = "
                              f"{clfs[0].get_n_label(0)!r}")
                break  # 只看第一个对象
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="输入视频 (MP4)")
    ap.add_argument("--dump-schema", action="store_true",
                    help="只打印字段名 schema")
    ap.add_argument("--dump-frame", type=int, default=5,
                    help="转储前 N 帧的完整字段值")
    ap.add_argument("--full", action="store_true",
                    help="跑完整两阶段流水线(person+helmet+harness_cls+vest_cls)，"
                         "转储二级分类器元数据 (classifier_items/get_n_label)")
    args = ap.parse_args()

    pgie = _anchor_ini_config(DEPLOY_ROOT / "generated" / "person" / "pgie_config.txt",
                              base=DEPLOY_ROOT)
    uri = "file://" + os.path.abspath(args.file)

    p = Pipeline("explore-metadata")
    p.add("nvurisrcbin", "src", {"uri": uri})
    p.add("nvstreammux", "mux", {"batch-size": 1, "width": 1920, "height": 1080,
                                  "batched-push-timeout": 33000})
    p.add("nvinfer", "pgie", {"config-file-path": pgie})
    tail = "pgie"
    if args.full:
        for name in ("helmet", "harness_cls", "vest_cls"):
            prefix = "sgie" if name.endswith("_cls") else "pgie"
            cfg = f"generated/{name}/{prefix}_config.txt"
            p.add("nvinfer", name,
                  {"config-file-path": _anchor_ini_config(DEPLOY_ROOT / cfg, base=DEPLOY_ROOT)})
            tail = name
    p.add("fakesink", "sink")
    p.link(("src", "mux"), ("", "sink_%u"))
    chain = ["pgie", "helmet", "harness_cls", "vest_cls"] if args.full else ["pgie"]
    p.link("mux", *chain, "sink")

    if args.dump_schema:
        probe = SchemaDumper()
    else:
        probe = MetadataDumper(args.dump_frame)
    p.attach(tail, Probe("metadata-dumper", probe))
    p.start().wait()


if __name__ == "__main__":
    proc = Process(target=main)
    proc.start()
    proc.join()
