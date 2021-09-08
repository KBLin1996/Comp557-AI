main(int numArgs, char *arguments[])
{
  if (numArgs != 4) // Usage is wrong, four arguments needed
  {
    cerr << "Usage: " << arguments[0] << " infile infile outfile" << endl;
    return (-1);
  }

  ifstream textFile(arguments[1], ios::in);
  if (!textFile)
  {
    cerr << "Can't find input file " << arguments[1] << endl;
    return (-1);
  }

  ifstream patternsFile(arguments[2], ios::in);
  if (!patternsFile)
  {
    cerr << "Can't find input file " << arguments[2] << endl;
    textFile.close();
    return (-1);
  }

  ofstream outputFile(arguments[3], ios::out);
  if (!outputFile)
  {
    cerr << "Can't create output file " << arguments[3] << endl;
    textFile.close();
    patternsFile.close();
    return (-1);
  }

  CHeap<CString> heap(50);
  getPatterns(patternsFile, &heap);
  matchPatterns(textFile, outputFile, &heap);

  textFile.close();
  patternsFile.close();
  outputFile.close();
}
