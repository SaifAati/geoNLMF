// -*- c++ -*-
// Saif Aati
// saif@caltech.edu

#ifndef geoNLMFLib_TYPECODE_H
#define geoNLMFLib_TYPECODE_H

namespace geoNLMFLib {

  enum DataTypeCode {
    UCHAR,
    USHORT,
    INT,
    UINT,
    FLOAT,
    DOUBLE,
  };


  template <int typecode>
  struct DataType {
    typedef unsigned char type;
  };

  template <>
  struct DataType <UCHAR> {
    typedef unsigned char type;
  };

  template <>
  struct DataType <USHORT> {
    typedef unsigned short type;
  };

  template <>
  struct DataType <INT> {
    typedef int type;
  };

  template <>
  struct DataType <UINT> {
    typedef unsigned int type;
  };

  template <>
  struct DataType <FLOAT> {
    typedef float type;
  };

  template <>
  struct DataType <DOUBLE>
  {
    typedef double type;
  };

}

#endif
