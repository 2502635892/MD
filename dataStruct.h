
typedef struct AtomInfo
{
  double mass;
  double position[Dim];
  double velocity[Dim];
  double force[Dim];
  int    atom_id;
} AtomInfo; // atom information


typedef struct TableInfo
{
  double L_cell[Dim];
  int cell_num[Dim];
  int Total_cellNUM;
} TableInfo; // cell table list information


typedef struct deProcess
{
  int x;
  int y; 
  int z;
} deProcess;

typedef struct seg
{
  int low;
  int high;
} seg; 

typedef struct subDomain
{
  seg X;
  seg Y;
  seg Z;
} subDomain; 



