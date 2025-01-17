package com.example.next;


import android.content.Context;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

public class BirthdaybookAdapter extends RecyclerView.Adapter<BirthdaybookAdapter.ViewHolder>
{
    private Fragment1 fragment1;
    private List<BirthdaybookData.ResultDTO> mDataBeanList = new ArrayList<>();
    //列表限制，在Fragment中调用setlistData时只能传列表，储存正常数据在ResultDTO中

    public void setListData(List<BirthdaybookData.ResultDTO> list) {
        this.mDataBeanList = list;
        Log.d("lx1", "list: " + list);
        //一定要调用
        notifyDataSetChanged();

    }
    public BirthdaybookAdapter(Fragment1 fragment1) {

        this.fragment1 = fragment1;
    }
    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType)
    {
        //绑定视图

        View view =
                LayoutInflater.from(parent.getContext()).inflate(R.layout.item_rv2, parent,
                        false);
        return new ViewHolder(view);
    }
    //对ViewHolder 中初始化好的控件进⾏数据绑定

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position)
    {
        BirthdaybookData.ResultDTO resultDTO = mDataBeanList.get(position);
        holder.titleTextView.setText(resultDTO.getTitle());
        holder.friendTextView.setText(resultDTO.getFriend());
        holder.lucky_numTextView.setText(resultDTO.getLucky_num());
        holder.businessTextView.setText(resultDTO.getBusiness());
        holder.in_loveTextView.setText(resultDTO.getIn_love());
        holder.moneyTextView.setText(resultDTO.getMoney());
        holder.BirthdayTextView.setText(resultDTO.getBirthday());
        holder.natureTextView.setText(resultDTO.getNature());
        //     @@@@@ here @@@@ 😉
    }

    //这个⽅法返回总共要⽣产多少个 View的数量
    @Override
    public int getItemCount() {
        return mDataBeanList.size();
    }

    // ViewHolder class
    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView titleTextView;
        TextView BirthdayTextView;
        TextView natureTextView;
        TextView moneyTextView;
        TextView businessTextView;
        TextView lucky_numTextView;
        TextView in_loveTextView;
        TextView friendTextView;

        public ViewHolder(@NonNull View itemView)
        {
            super(itemView);
            titleTextView = itemView.findViewById(R.id.item_title);
            BirthdayTextView=itemView.findViewById(R.id.item_birthday);
            natureTextView=itemView.findViewById(R.id.item_nature);
            moneyTextView=itemView.findViewById(R.id.item_money);
            businessTextView=itemView.findViewById(R.id.item_bussiness);
            lucky_numTextView=itemView.findViewById(R.id.item_lucky_num);
            in_loveTextView=itemView.findViewById(R.id.item_in_love);
            friendTextView=itemView.findViewById(R.id.item_friend);
        }
    }
}
