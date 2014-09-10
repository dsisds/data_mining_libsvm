#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "libxml++/libxml++.h" 
#include <libxml++/nodes/node.h>
#include <stdio.h>
#include <stdlib.h>
#include "xmlreader.h"
#include <map>

using namespace std;

int parsexml(string xmlfile, map<string, vector<vector<int> > >& gt){
    xmlpp::DomParser parser;
    parser.parse_file( xmlfile );
    if(!parser){
        cout << "can't open xmlfile:" << xmlfile << endl;
        return 1;
    }
    xmlpp::Node* pNode = parser.get_document() -> get_root_node();
 //   cout << "root element is \"" << pNode -> get_name() << "\"" << endl;
    xmlpp::Node::NodeList node_list = pNode -> get_children();
    xmlpp::Node::NodeList::iterator iter = node_list.begin();
    for(; iter != node_list.end(); ++iter ){
        xmlpp::Element * content = dynamic_cast<xmlpp::Element *>(*iter);
        if ( NULL != content ){
//            cout << (*iter) -> get_name() << endl;
            if((*iter)->get_name() == "object"){
                xmlpp::Node::NodeList object = (*iter)->get_children();
                xmlpp::Node::NodeList::iterator it = object.begin();
                vector<int> bndbox(4,0);
                string con;
                bool isvalid = true;
                for(; it != object.end(); ++it){
                    xmlpp::Element* ob = dynamic_cast<xmlpp::Element *>(*it);
                    if(ob!=NULL){
                        string ele_name = (*it)->get_name();
                        if(ele_name == "name"){
                            //cout << "cls:" << (*it)->get_name() << " " << ob->get_child_text()->get_content() << " ";
                           con=ob->get_child_text()->get_content();

                        }else if(ele_name =="bndbox"){
                            xmlpp::Node::NodeList bbox = (*it)->get_children();
                            xmlpp::Node::NodeList::iterator coord = bbox.begin();
                            for(; coord != bbox.end(); ++coord){
                                xmlpp::Element* c = dynamic_cast<xmlpp::Element *>(*coord);
                                if(c==NULL)
                                    continue;
                                string cn = (c)->get_name();
                                if(cn == "xmin"){
                                    bndbox[0] = atoi((c)->get_child_text()->get_content().c_str())-1;
                                 //   cout << "xmin:" << (c)->get_child_text()->get_content() << " ";
                                }else if(cn == "xmax"){
                                    bndbox[2] = atoi((c)->get_child_text()->get_content().c_str())-1;
                                   // cout << "xmax:" << (c)->get_child_text()->get_content() << " ";
                                }else if(cn == "ymin"){
                                    bndbox[1] = atoi((c)->get_child_text()->get_content().c_str())-1;
                                    //cout << "ymin:" << (c)->get_child_text()->get_content() << " ";
                                }else if(cn == "ymax"){
                                    bndbox[3] = atoi((c)->get_child_text()->get_content().c_str())-1;
                                    //cout << "ymax:" << (c)->get_child_text()->get_content() << " ";
                                }
                            }
                            //cout << endl;
                        }
                    }
                }
                if(gt.find(con)==gt.end()){
                    vector<vector<int> > tmp_box;
                    tmp_box.push_back(bndbox);
                    gt.insert(make_pair(con, tmp_box));
                }else{
                    gt[con].push_back(bndbox);
                }
            }
        }
    }
    return 0;
}
/*
int main(int argc, char** argv){
    vector< vector<int> > gt;
    int s= parsexml(argv[1], gt, "cow");
    cout << "gt:" << gt.size() << endl;
}*/

