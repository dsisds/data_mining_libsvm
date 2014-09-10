#ifndef XMLREADER_H_
#define XMLREADER_H_

#include <string>
#include <vector>
#include <map>
using namespace std;

int parsexml(string xmlfile, map<string, vector<vector<int> > >& gt);

#endif
