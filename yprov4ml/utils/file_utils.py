
import os
import prov.model as prov

from yprov4ml.constants import PROV4ML_DATA

def save_prov_file(
        doc : prov.ProvDocument,
        prov_file : str,
        create_graph : bool =False, 
        create_svg : bool =False
    ) -> None:
    """
    Save the provenance document to a file.

    Parameters:
    -----------
    doc : prov.ProvDocument
        The provenance document to save.
    prov_file : str
        The path to the file where the provenance document will be saved.
    create_graph : bool 
        A flag to indicate if a graph should be created. Defaults to False.
    create_svg : bool
        A flag to indicate if an SVG should be created. Defaults to False.
    
    Returns:
        None
    """

    with open(prov_file, 'w') as prov_graph:
        doc.serialize(prov_graph)

    if create_svg and not create_graph:
        raise ValueError("Cannot create SVG without creating the graph.")

    if create_graph:
        dot_filename = os.path.basename(prov_file).replace(".json", ".dot")
        path_dot = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, dot_filename)
        with open(path_dot, 'w') as prov_dot:
            prov_dot.write(custom_prov_to_dot(doc).to_string())

    if create_svg:
        svg_filename = os.path.basename(prov_file).replace(".json", ".svg")
        path_svg = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, svg_filename)
        os.system(f"dot -Tsvg {path_dot} > {path_svg}")


from prov.model import (
    ProvException,
    Identifier,
    PROV_ATTRIBUTE_QNAMES,
    datetime,
)
from prov.graph import INFERRED_ELEMENT_CLASS
from prov.dot import ANNOTATION_STYLE, ANNOTATION_LINK_STYLE, ANNOTATION_START_ROW, ANNOTATION_ROW_TEMPLATE, ANNOTATION_END_ROW, DOT_PROV_STYLE, GENERIC_NODE_STYLE, sorted_attributes
from xml.sax.saxutils import escape
from datetime import datetime
import pydot

def custom_prov_to_dot(
    bundle,
    show_nary=True,
    use_labels=False,
    direction="RL",
    show_element_attributes=True,
    show_relation_attributes=True,
):
    """
    Convert a provenance bundle/document into a DOT graphical representation.

    :param bundle: The provenance bundle/document to be converted.
    :type bundle: :class:`ProvBundle`
    :param show_nary: shows all elements in n-ary relations.
    :type show_nary: bool
    :param use_labels: uses the prov:label property of an element as its name (instead of its identifier).
    :type use_labels: bool
    :param direction: specifies the direction of the graph. Valid values are "BT" (default), "TB", "LR", "RL".
    :param show_element_attributes: shows attributes of elements.
    :type show_element_attributes: bool
    :param show_relation_attributes: shows attributes of relations.
    :type show_relation_attributes: bool
    :returns:  :class:`pydot.Dot` -- the Dot object.
    """
    if direction not in {"BT", "TB", "LR", "RL"}:
        # Invalid direction is provided
        direction = "BT"  # reset it to the default value
    maindot = pydot.Dot(graph_type="digraph", rankdir=direction, charset="utf-8")

    node_map = {}
    count = [0, 0, 0, 0]  # counters for node ids

    def _bundle_to_dot(dot, bundle):
        def _attach_attribute_annotation(node, record):
            # Adding a node to show all attributes
            attributes = []
                # Truncate the value if it is too long
            for attr_name, value in record.attributes: 
                if attr_name not in PROV_ATTRIBUTE_QNAMES:
                    try: 
                        if len(value) > 100: 
                            attributes.append((attr_name, str(value[:100]) + "..."))
                        else: 
                            attributes.append((attr_name, value))
                    except TypeError: 
                        attributes.append((attr_name, value))

            if not attributes or len(attributes) == 0:
                return  # No attribute to display

            # Sort the attributes.
            attributes = sorted_attributes(record.get_type(), attributes)

            ann_rows = [ANNOTATION_START_ROW]
            ann_rows.extend(
                ANNOTATION_ROW_TEMPLATE
                % (
                    attr.uri,
                    escape(str(attr)),
                    ' href="%s"' % value.uri if isinstance(value, Identifier) else "",
                    escape(
                        str(value)
                        if not isinstance(value, datetime)
                        else str(value.isoformat())
                    ),
                )
                for attr, value in attributes
            )
            ann_rows.append(ANNOTATION_END_ROW)
            count[3] += 1
            annotations = pydot.Node(
                "ann%d" % count[3], label="\n".join(ann_rows), **ANNOTATION_STYLE
            )
            dot.add_node(annotations)
            dot.add_edge(pydot.Edge(annotations, node, **ANNOTATION_LINK_STYLE))

        def _add_bundle(bundle):
            count[2] += 1
            subdot = pydot.Cluster(
                graph_name="c%d" % count[2], URL=f'"{bundle.identifier.uri}"'
            )
            if use_labels:
                if bundle.label == bundle.identifier:
                    bundle_label = f'"{bundle.label}"'
                else:
                    # Fancier label if both are different. The label will be the main
                    # node text, whereas the identifier will be a kind of subtitle.
                    bundle_label = (
                        f"<{bundle.label}<br />"
                        f'<font color="#333333" point-size="10">'
                        f'{bundle.identifier}</font>>'
                    )
                subdot.set_label(f'"{bundle_label}"')
            else:
                subdot.set_label('"%s"' % str(bundle.identifier))
            _bundle_to_dot(subdot, bundle)
            dot.add_subgraph(subdot)
            return subdot

        def _add_node(record):
            count[0] += 1
            node_id = "n%d" % count[0]
            if use_labels:
                if record.label == record.identifier:
                    node_label = f'"{record.label}"'
                else:
                    # Fancier label if both are different. The label will be
                    # the main node text, whereas the identifier will be a
                    # kind of subtitle.
                    node_label = (
                        f"<{record.label}<br />"
                        f'<font color="#333333" point-size="10">'
                        f'{record.identifier}</font>>'
                    )
            else:
                node_label = f'"{record.identifier}"'

            uri = record.identifier.uri
            style = DOT_PROV_STYLE[record.get_type()]
            node = pydot.Node(node_id, label=node_label, URL='"%s"' % uri, **style)
            node_map[uri] = node
            dot.add_node(node)

            if show_element_attributes:
                _attach_attribute_annotation(node, rec)
            return node

        def _add_generic_node(qname, prov_type=None):
            count[0] += 1
            node_id = "n%d" % count[0]
            node_label = f'"{qname}"'

            uri = qname.uri
            style = GENERIC_NODE_STYLE[prov_type] if prov_type else DOT_PROV_STYLE[0]
            node = pydot.Node(node_id, label=node_label, URL='"%s"' % uri, **style)
            node_map[uri] = node
            dot.add_node(node)
            return node

        def _get_bnode():
            count[1] += 1
            bnode_id = "b%d" % count[1]
            bnode = pydot.Node(bnode_id, label='""', shape="point", color="gray")
            dot.add_node(bnode)
            return bnode

        def _get_node(qname, prov_type=None):
            if qname is None:
                return _get_bnode()
            uri = qname.uri
            if uri not in node_map:
                _add_generic_node(qname, prov_type)
            return node_map[uri]

        records = bundle.get_records()
        relations = []
        for rec in records:
            if rec.is_element():
                _add_node(rec)
            else:
                # Saving the relations for later processing
                relations.append(rec)

        if not bundle.is_bundle():
            for bundle in bundle.bundles:
                _add_bundle(bundle)

        for rec in relations:
            args = rec.args
            # skipping empty records
            if not args:
                continue
            # picking element nodes
            attr_names, nodes = zip(
                *(
                    (attr_name, value)
                    for attr_name, value in rec.formal_attributes
                    if attr_name in PROV_ATTRIBUTE_QNAMES
                )
            )
            inferred_types = list(map(INFERRED_ELEMENT_CLASS.get, attr_names))
            other_attributes = [
                (attr_name, value)
                for attr_name, value in rec.attributes
                if attr_name not in PROV_ATTRIBUTE_QNAMES
            ]
            add_attribute_annotation = show_relation_attributes and other_attributes
            add_nary_elements = len(nodes) > 2 and show_nary
            style = DOT_PROV_STYLE[rec.get_type()]
            if len(nodes) < 2:  # too few elements for a relation?
                continue  # cannot draw this

            if add_nary_elements or add_attribute_annotation:
                # a blank node for n-ary relations or the attribute annotation
                bnode = _get_bnode()

                # the first segment
                dot.add_edge(
                    pydot.Edge(
                        _get_node(nodes[0], inferred_types[0]),
                        bnode,
                        arrowhead="none",
                        **style,
                    )
                )
                style = dict(style)  # copy the style
                del style["label"]  # not showing label in the second segment
                # the second segment
                dot.add_edge(
                    pydot.Edge(bnode, _get_node(nodes[1], inferred_types[1]), **style)
                )
                if add_nary_elements:
                    style["color"] = "gray"  # all remaining segment to be gray
                    style["fontcolor"] = "dimgray"  # text in darker gray
                    for attr_name, node, inferred_type in zip(
                        attr_names[2:], nodes[2:], inferred_types[2:]
                    ):
                        if node is not None:
                            style["label"] = attr_name.localpart
                            dot.add_edge(
                                pydot.Edge(
                                    bnode, _get_node(node, inferred_type), **style
                                )
                            )
                if add_attribute_annotation:
                    _attach_attribute_annotation(bnode, rec)
            else:
                # show a simple binary relations with no annotation
                dot.add_edge(
                    pydot.Edge(
                        _get_node(nodes[0], inferred_types[0]),
                        _get_node(nodes[1], inferred_types[1]),
                        **style,
                    )
                )

    try:
        unified = bundle.unified()
    except ProvException:
        # Could not unify this bundle
        # try the original document anyway
        unified = bundle

    _bundle_to_dot(maindot, unified)
    return maindot
