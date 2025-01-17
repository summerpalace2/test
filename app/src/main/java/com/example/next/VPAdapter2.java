package com.example.next;


import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager2.adapter.FragmentStateAdapter;

import java.util.ArrayList;

public class VPAdapter2 extends FragmentStateAdapter
{
    private final ArrayList<FragmentInterface> fragments;
    public VPAdapter2(@NonNull FragmentActivity fragmentActivity,
                      ArrayList<FragmentInterface> fragments)
    {
        super(fragmentActivity);
        this.fragments = fragments;
    }

    @NonNull
    @Override //从提供的Fragment数据源中拿数据
    public Fragment createFragment(int position) {
        return fragments.get(position).back();
    }
    @Override
    //返回Fragment的个数
    public int getItemCount() {
        return fragments.size();
    }
}
