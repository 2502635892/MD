void GetTableInfo(TableInfo &T_Info)
{
  double Box_Lx = L_x * a_x;
  double Box_Ly = L_y * a_y;
  double Box_Lz = L_z * a_z; // the simulation area Box size
  
  T_Info.cell_num[0] = (int)(Box_Lx / RCUT);
  T_Info.cell_num[1] = (int)(Box_Ly / RCUT);
  T_Info.cell_num[2] = (int)(Box_Lz / RCUT); // the cell divide on x ,y ,z direction & the number of cell bins

  T_Info.L_cell[0] = Box_Lx / (double)T_Info.cell_num[0];
  T_Info.L_cell[1] = Box_Ly / (double)T_Info.cell_num[1];
  T_Info.L_cell[2] = Box_Lz / (double)T_Info.cell_num[2]; // get the cell length

  T_Info.Total_cellNUM = T_Info.cell_num[0] * T_Info.cell_num[1] * T_Info.cell_num[2];
}



//-------------------------make the segment partition------------
void segPartiton(seg *segPoint,int segCount,int n)
{
      for(int i=0 ;i < segCount; i++)
    {
      segPoint[i].low = BLOCK_LOW(i,segCount,n);
      segPoint[i].high = BLOCK_HIGH(i,segCount,n);
	 }

}


int GetCellSubDomainId(int cx,int cy,int cz,TableInfo T_Info,deProcess P)
{
    int subDomain_id = -1;

    int Xcellnum = T_Info.cell_num[0];
	int Ycellnum = T_Info.cell_num[1];
	int Zcellnum = T_Info.cell_num[2];

	int XsubId = BLOCK_OWNER(cx,P.x,Xcellnum);
	int YsubId = BLOCK_OWNER(cy,P.y,Ycellnum);
	int ZsubId = BLOCK_OWNER(cz,P.z,Zcellnum);

    subDomain_id = XsubId + YsubId * P.x + ZsubId * P.x * P.y;
    return subDomain_id;

}



void check_GetCellSubDomainId(TableInfo T_Info,subDomain *subdomain,deProcess P)
{
  int count = 0;
  for(int i=0;i<T_Info.cell_num[2];i++ )
  {
      for(int j = 0;j< T_Info.cell_num[1];j++)
      { 
          for(int k=0;k<T_Info.cell_num[0];k++)
          {
              int id =  GetCellSubDomainId( k , j , i ,T_Info, P);
              bool b1 = ( k>= subdomain[id].X.low) && (k<=subdomain[id].X.high);
              bool b2 = ( j>= subdomain[id].Y.low) && (j<=subdomain[id].Y.high);
              bool b3 = ( i>= subdomain[id].Z.low) && (i<=subdomain[id].Z.high);
              bool b = ( (b1&&b2) && b3);
              if(b)
              {
                 count++;
              }else
              {
                  printf("error!!\n");
              }

      }

  }

}

if(count == T_Info.cell_num[0]*T_Info.cell_num[1]*T_Info.cell_num[2])
  printf("check is ok\n");


}